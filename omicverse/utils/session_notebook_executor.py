"""
Session-Based Notebook Executor for OmicVerse Agent

This module provides persistent notebook session management for executing
generated code across multiple prompts while maintaining context and state.

Key Features:
- Session-based execution (variables persist across prompts)
- Automatic session restart after N prompts (prevents memory bloat)
- Conda environment detection and kernel management
- Robust error handling and recovery
- Notebook persistence for debugging and reproducibility
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from queue import Empty
import warnings


class KernelNotFoundError(Exception):
    """Raised when required Jupyter kernel is not found in the environment."""
    pass


class SessionNotebookExecutor:
    """
    Manages a persistent Jupyter notebook session for multiple agent executions.

    This executor maintains a single notebook kernel across multiple code execution
    requests, allowing variables, imports, and state to persist between prompts.
    Sessions automatically restart after a configurable number of prompts to prevent
    memory bloat.

    Parameters
    ----------
    max_prompts_per_session : int, default 5
        Maximum prompts before restarting session (prevents memory bloat)
    storage_dir : Optional[Path], default None
        Directory to store session notebooks. Defaults to ~/.ovagent/sessions
    keep_notebooks : bool, default True
        Whether to keep session notebooks after execution
    timeout : int, default 600
        Execution timeout in seconds
    strict_kernel_validation : bool, default True
        If True, raise KernelNotFoundError if kernel not found.
        If False, fall back to 'python3' kernel with warning.

    Attributes
    ----------
    conda_env : Optional[str]
        Detected conda environment name
    kernel_name : str
        Jupyter kernel name to use
    current_session : Optional[Dict]
        Current session state information
    session_prompt_count : int
        Number of prompts executed in current session
    session_history : List[Dict]
        History of all archived sessions

    Examples
    --------
    >>> from omicverse.utils.session_notebook_executor import SessionNotebookExecutor
    >>> executor = SessionNotebookExecutor(max_prompts_per_session=5)
    >>> result = executor.execute(code, adata, execution_id="prompt_1")
    """

    def __init__(
        self,
        max_prompts_per_session: int = 5,
        storage_dir: Optional[Path] = None,
        keep_notebooks: bool = True,
        timeout: int = 600,
        strict_kernel_validation: bool = True
    ):
        """Initialize SessionNotebookExecutor with configuration."""
        self.max_prompts_per_session = max_prompts_per_session
        from ._ovagent_paths import ovagent_home
        self.storage_dir = storage_dir or (ovagent_home() / "sessions")
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.keep_notebooks = keep_notebooks
        self.timeout = timeout
        self.strict_kernel_validation = strict_kernel_validation

        # Session state
        self.current_session: Optional[Dict[str, Any]] = None
        self.session_prompt_count: int = 0
        self.session_history: List[Dict[str, Any]] = []

        # Conda environment detection
        self.conda_env = self._detect_conda_environment()
        self.kernel_name = self._get_kernel_name()

        # Ensure kernel is installed
        self._ensure_kernel_installed(strict=self.strict_kernel_validation)

    # ===================================================================
    # Conda Environment Detection
    # ===================================================================

    def _detect_conda_environment(self) -> Optional[str]:
        """
        Detect current conda environment name.

        Tries multiple methods to detect the conda environment:
        1. CONDA_DEFAULT_ENV environment variable
        2. Parse CONDA_PREFIX environment variable
        3. Check sys.prefix for conda/anaconda paths

        Returns
        -------
        Optional[str]
            Conda environment name, or None if not in conda environment
        """
        # Method 1: CONDA_DEFAULT_ENV environment variable
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            return conda_env

        # Method 2: Parse conda prefix
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            return Path(conda_prefix).name

        # Method 3: Check if running in conda (sys.prefix)
        if 'conda' in sys.prefix.lower() or 'anaconda' in sys.prefix.lower():
            return Path(sys.prefix).name

        return None

    def _get_kernel_name(self) -> str:
        """
        Get kernel name for current conda environment.

        Returns
        -------
        str
            Kernel name to use for execution
        """
        if self.conda_env:
            # Standard conda kernel naming convention
            return f'conda-env-{self.conda_env}-py'
        return 'python3'  # Fallback to default Python kernel

    # ===================================================================
    # Kernel Installation Check
    # ===================================================================

    def _ensure_kernel_installed(self, strict: bool = True) -> bool:
        """
        Verify kernel exists in conda environment.

        This method checks if the required Jupyter kernel is installed and
        available for use. In strict mode, it fails fast if the kernel is
        not found. In non-strict mode, it falls back to the default python3
        kernel with a warning.

        Parameters
        ----------
        strict : bool, default True
            If True, raise KernelNotFoundError if kernel not found.
            If False, fall back to 'python3' kernel with warning.

        Returns
        -------
        bool
            True if kernel found, False if using fallback

        Raises
        ------
        KernelNotFoundError
            If strict=True and no matching kernel is found.

        Examples
        --------
        >>> executor = SessionNotebookExecutor(strict_kernel_validation=True)
        Traceback (most recent call last):
            ...
        KernelNotFoundError: [FAIL] Kernel not found for conda environment 'myenv'
        """
        try:
            from jupyter_client.kernelspec import KernelSpecManager
        except ImportError:
            if strict:
                raise KernelNotFoundError(
                    "[FAIL] jupyter_client not installed.\n"
                    "Install with: pip install jupyter-client ipykernel\n"
                    "Or: conda install jupyter ipykernel"
                )
            else:
                warnings.warn(
                    "jupyter_client not installed. Falling back to in-process execution.\n"
                    "Install with: pip install omicverse[agent]"
                )
                return False

        ksm = KernelSpecManager()
        available_kernels = ksm.find_kernel_specs()

        # Look for environment-specific kernel
        possible_names = [
            f'conda-env-{self.conda_env}-py',
            self.conda_env,
            'python3'
        ]

        for name in possible_names:
            if name and name in available_kernels:
                self.kernel_name = name
                print(f"✓ [OK] Using kernel: {name} (conda env: {self.conda_env or 'default'})")
                return True

        # Kernel not found - fail fast or fallback based on strict mode
        if strict:
            available_list = '\n'.join(f"  - {k}" for k in available_kernels.keys())
            error_msg = (
                f"[FAIL] Kernel not found for conda environment '{self.conda_env}'\n\n"
                f"Available kernels:\n{available_list}\n\n"
                f"To fix this, run:\n"
                f"  python -m ipykernel install --user --name {self.conda_env}\n\n"
                f"Or use an existing kernel:\n"
                f"  agent = ov.Agent(kernel='{list(available_kernels.keys())[0] if available_kernels else 'python3'}')\n\n"
                f"Or disable strict validation (not recommended):\n"
                f"  agent = ov.Agent(strict_kernel_validation=False)"
            )
            raise KernelNotFoundError(error_msg)
        else:
            # Legacy fallback behavior (not recommended)
            print(f"⚠ [WARN] No kernel found for conda env '{self.conda_env}'")
            print(f"  [TIP] Install with: python -m ipykernel install --user --name {self.conda_env}")
            print(f"⚠ [WARN] Falling back to 'python3' kernel (may use different environment!)")
            self.kernel_name = 'python3'
            return False

    # ===================================================================
    # Kernel Lifecycle Management
    # ===================================================================

    def _wait_for_kernel_ready(self, kc, timeout: int = 30) -> bool:
        """
        Wait for kernel to be fully ready before executing code.

        Uses jupyter_client's kernel_info request to verify kernel is responsive.

        Parameters
        ----------
        kc : KernelClient
            Kernel client instance
        timeout : int, default 30
            Maximum time to wait in seconds

        Returns
        -------
        bool
            True if kernel is ready

        Raises
        ------
        TimeoutError
            If kernel fails to become ready within timeout
        """
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Kernel failed to become ready within {timeout}s")

            try:
                # Send kernel_info_request to check if kernel is ready
                kc.kernel_info()

                # Wait for kernel_info_reply
                try:
                    reply = kc.get_shell_msg(timeout=2.0)
                    if reply['msg_type'] == 'kernel_info_reply':
                        kernel_info = reply['content']
                        print(f"✓ [OK] Kernel ready: {kernel_info['language_info']['name']} "
                              f"v{kernel_info['language_info']['version']}")
                        return True
                except Empty:
                    # Kernel not ready yet, continue waiting
                    time.sleep(0.5)
                    continue

            except Exception:
                # Communication error, kernel might not be ready
                time.sleep(0.5)
                continue

    def _is_kernel_alive(self) -> bool:
        """
        Check if kernel is still running.

        Returns
        -------
        bool
            True if kernel is alive and responsive, False otherwise
        """
        if not self.current_session:
            return False

        km = self.current_session['kernel_manager']
        kc = self.current_session['kernel_client']

        try:
            # Check if kernel process is still running
            if not km.is_alive():
                return False

            # Send kernel_info_request and wait for the matching reply,
            # draining any stale messages that may be ahead in the queue.
            kc.kernel_info()
            deadline = time.time() + 5.0
            while time.time() < deadline:
                try:
                    reply = kc.get_shell_msg(timeout=0.5)
                    if reply['msg_type'] == 'kernel_info_reply':
                        return True
                except Exception:
                    break
            return False

        except Exception:
            return False

    def _interrupt_kernel(self):
        """
        Send interrupt signal to kernel (like Ctrl+C).

        This is useful for canceling long-running operations without
        restarting the entire kernel.
        """
        if not self.current_session:
            return

        km = self.current_session['kernel_manager']

        print("⚠ Interrupting kernel...")

        try:
            km.interrupt_kernel()
            time.sleep(0.5)  # Give kernel time to handle interrupt
            print("✓ [OK] Kernel interrupted")

        except Exception as e:
            print(f"✗ [FAIL] Failed to interrupt kernel: {e}")

    def _restart_kernel(self):
        """
        Restart kernel without losing session state.

        This clears the kernel's memory but keeps the kernel process
        and client connections alive.
        """
        if not self.current_session:
            return

        km = self.current_session['kernel_manager']
        old_kc = self.current_session['kernel_client']

        print("⚙ [RESTART] Restarting kernel...")

        # Stop channels
        try:
            old_kc.stop_channels()
        except Exception as e:
            print(f"⚠ [WARN] Failed to stop old kernel channels cleanly: {e}")

        # Restart kernel (keeps process alive, clears state)
        km.restart_kernel(now=True)

        # Reconnect with a fresh client; stale sockets are a common failure mode.
        kc = km.client()
        kc.start_channels()
        self.current_session['kernel_client'] = kc

        # Wait for kernel ready
        self._wait_for_kernel_ready(kc)

        print("✓ [OK] Kernel restarted successfully")

    def _shutdown_kernel(self):
        """
        Gracefully shutdown kernel.

        This performs a clean shutdown of the kernel process and
        closes all communication channels.
        """
        if not self.current_session:
            return

        km = self.current_session['kernel_manager']
        kc = self.current_session['kernel_client']

        print("⚙ Shutting down kernel...")

        try:
            # Stop message channels
            kc.stop_channels()

            # Shutdown kernel (graceful)
            km.shutdown_kernel(now=False, restart=False)

            # Wait for shutdown (with timeout)
            try:
                km.wait_for_ready(timeout=10)
            except:
                pass  # Ignore timeout errors during shutdown

        except Exception as e:
            # Force kill if graceful shutdown fails
            print(f"⚠ [WARN] Graceful shutdown failed, force killing: {e}")
            try:
                km.shutdown_kernel(now=True, restart=False)
            except:
                pass  # Ignore errors during force shutdown

        print("✓ [OK] Kernel shutdown complete")

    def _recover_from_kernel_failure(self) -> bool:
        """
        Attempt to recover from kernel crash or hang.

        This method tries multiple recovery strategies:
        1. Interrupt the kernel if it's hung
        2. Restart the kernel if interrupt fails
        3. Create a new session if restart fails

        Returns
        -------
        bool
            True if recovery succeeded, False otherwise
        """
        if not self.current_session:
            return False

        km = self.current_session['kernel_manager']
        # Check if kernel is truly dead
        if km.is_alive():
            # Kernel process is alive, try interrupting first
            print("⚠ Kernel unresponsive, attempting interrupt...")
            self._interrupt_kernel()

            # Test if interrupt worked
            time.sleep(1)
            if self._is_kernel_alive():
                print("✓ [OK] Kernel recovered via interrupt")
                return True

        # Kernel dead or interrupt failed, try restart
        print("⚙ Kernel dead, attempting restart...")

        try:
            self._restart_kernel()
            kc = self.current_session['kernel_client']

            # Re-initialize session imports
            init_code = """
import omicverse as ov
import pandas as pd
import numpy as np
print("✓ [OK] Session re-initialized after recovery")
"""
            self._execute_code_in_kernel(init_code, kc)

            print("✓ [OK] Kernel recovered via restart")
            return True

        except Exception as e:
            print(f"✗ [FAIL] Kernel recovery failed: {e}")
            print("⚙ [RESTART] Creating new session...")

            # Archive failed session
            self._archive_current_session()

            # Start fresh session
            self._start_new_session()

            return True  # Recovered by creating new session

    # ===================================================================
    # Session Lifecycle Management
    # ===================================================================

    def _should_start_new_session(self) -> bool:
        """
        Check if new session needed.

        Returns
        -------
        bool
            True if new session should be started, False otherwise
        """
        # No session exists
        if self.current_session is None:
            return True

        # Session limit reached
        if self.session_prompt_count >= self.max_prompts_per_session:
            print(f"⚙ = Session limit reached ({self.max_prompts_per_session} prompts)")
            print(f"⚙ = Starting fresh session...")
            self._archive_current_session()
            return True

        # Kernel crashed
        if not self._is_kernel_alive():
            print(f"⚠ Kernel crashed, restarting session...")
            self._archive_current_session()
            return True

        return False

    def _start_new_session(self):
        """
        Initialize new notebook session.

        Creates a new kernel, initializes imports, and sets up
        the session directory structure.
        """
        try:
            from jupyter_client import KernelManager
        except ImportError:
            raise RuntimeError(
                "jupyter_client not installed. "
                "Install with: pip install jupyter-client ipykernel"
            )

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.storage_dir / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True, parents=True)

        print(f"⚙ Starting new session: {session_id}")

        # Start kernel
        km = KernelManager(kernel_name=self.kernel_name)
        km.start_kernel()
        kc = km.client()
        kc.start_channels()

        # Wait for kernel ready
        self._wait_for_kernel_ready(kc)

        # Initialize session imports
        init_code = """
import matplotlib
matplotlib.use('Agg')
import omicverse as ov
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
print("✓ Session initialized")
"""
        try:
            outputs = self._execute_code_in_kernel(init_code, kc)
            if outputs['stdout']:
                print(''.join(outputs['stdout']))
        except Exception as e:
            print(f"⚠ [WARN] Session initialization warning: {e}")

        # Create notebook structure
        nb = self._create_session_notebook(session_id)
        nb_path = session_dir / "session.ipynb"

        # Save initial notebook
        try:
            import nbformat
            with open(nb_path, 'w') as f:
                nbformat.write(nb, f)
        except ImportError:
            print(f"⚠ [WARN] nbformat not installed, notebook will not be saved")
            nb = None

        # Store session info
        self.current_session = {
            'session_id': session_id,
            'session_dir': session_dir,
            'notebook_path': nb_path,
            'kernel_manager': km,
            'kernel_client': kc,
            'notebook': nb,
            'start_time': datetime.now(),
            'executions': []
        }

        self.session_prompt_count = 0

        print(f"✓ New session started: {session_id}")
        print(f"  Conda environment: {self.conda_env or 'default'}")
        print(f"  Session notebook: {nb_path}")

    def _archive_current_session(self):
        """
        Archive current session before starting new one.

        Saves the notebook, shuts down the kernel, and stores
        session metadata in history.
        """
        if self.current_session is None:
            return

        print(f"⚙ Archiving session: {self.current_session['session_id']}")

        # Shutdown kernel
        self._shutdown_kernel()

        # Save final notebook
        nb_path = self.current_session['notebook_path']
        if self.current_session['notebook'] is not None:
            try:
                import nbformat
                with open(nb_path, 'w') as f:
                    nbformat.write(self.current_session['notebook'], f)
            except Exception as e:
                print(f"⚠ [WARN] Failed to save notebook: {e}")

        # Add to history
        self.session_history.append({
            'session_id': self.current_session['session_id'],
            'notebook_path': str(nb_path),
            'prompt_count': self.session_prompt_count,
            'start_time': self.current_session['start_time'],
            'end_time': datetime.now(),
            'executions': self.current_session['executions']
        })

        print(f"✓ Session archived: {nb_path}")
        print(f"  Total prompts in session: {self.session_prompt_count}")

        # Cleanup temp files if not keeping notebooks
        if not self.keep_notebooks:
            import shutil
            session_dir = self.current_session['session_dir']
            try:
                shutil.rmtree(session_dir)
                print(f"  Cleaned up session directory")
            except Exception as e:
                print(f"⚠ [WARN] Failed to cleanup session directory: {e}")

    # ===================================================================
    # Notebook Structure & Persistence
    # ===================================================================

    def _create_session_notebook(self, session_id: str):
        """
        Create initial session notebook structure.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        nbformat.NotebookNode or None
            Notebook object, or None if nbformat not available
        """
        try:
            import nbformat
            from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
        except ImportError:
            return None

        nb = new_notebook()

        nb.metadata = {
            'kernelspec': {
                'display_name': f'Python ({self.conda_env or "default"})',
                'language': 'python',
                'name': self.kernel_name
            },
            'conda_env': self.conda_env,
            'session_id': session_id,
            'max_prompts': self.max_prompts_per_session
        }

        nb.cells = [
            new_markdown_cell(f"# OmicVerse Agent Session\n\n"
                            f"**Session ID:** `{session_id}`\n"
                            f"**Conda Environment:** `{self.conda_env or 'default'}`\n"
                            f"**Max Prompts:** {self.max_prompts_per_session}\n"
                            f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),

            new_markdown_cell("## Session Initialization"),
            new_code_cell("""import omicverse as ov
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')""")
        ]

        return nb

    def _append_to_session_notebook(self, code: str, outputs: Dict[str, Any]):
        """
        Append executed code to session notebook.

        Parameters
        ----------
        code : str
            Code that was executed
        outputs : dict
            Execution outputs from _execute_code_in_kernel
        """
        if self.current_session['notebook'] is None:
            return  # nbformat not available

        try:
            import nbformat
            from nbformat.v4 import new_markdown_cell, new_code_cell
        except ImportError:
            return

        _node = nbformat.from_dict

        nb = self.current_session['notebook']

        # Add markdown separator
        nb.cells.append(
            new_markdown_cell(f"## Prompt {self.session_prompt_count + 1}")
        )

        # Add code cell with outputs
        code_cell = new_code_cell(code)

        # Add stdout outputs
        if outputs['stdout']:
            code_cell.outputs.append(_node({
                'output_type': 'stream',
                'name': 'stdout',
                'text': ''.join(outputs['stdout'])
            }))

        # Add stderr outputs
        if outputs['stderr']:
            code_cell.outputs.append(_node({
                'output_type': 'stream',
                'name': 'stderr',
                'text': ''.join(outputs['stderr'])
            }))

        # Add error outputs
        if outputs['errors']:
            for error in outputs['errors']:
                code_cell.outputs.append(_node({
                    'output_type': 'error',
                    'ename': error['ename'],
                    'evalue': error['evalue'],
                    'traceback': error['traceback']
                }))

        # Add display data outputs
        if outputs['display_data']:
            for data in outputs['display_data']:
                code_cell.outputs.append(_node({
                    'output_type': 'display_data',
                    'data': data,
                    'metadata': {}
                }))

        nb.cells.append(code_cell)

        # Save notebook incrementally
        try:
            import nbformat
            with open(self.current_session['notebook_path'], 'w') as f:
                nbformat.write(nb, f)
        except Exception as e:
            print(f"⚠ [WARN] Failed to update notebook: {e}")

    # ===================================================================
    # Code Execution Pipeline
    # ===================================================================

    def _execute_code_in_kernel(self, code: str, kc, auto_recover: bool = True) -> Dict[str, Any]:
        """
        Execute code in kernel and capture outputs.

        This method handles the low-level jupyter_client communication protocol,
        capturing stdout, stderr, display data, and errors from the kernel.

        Parameters
        ----------
        code : str
            Python code to execute
        kc : KernelClient
            Kernel client instance
        auto_recover : bool, default True
            If True, attempt automatic recovery on timeout via interrupt -> restart

        Returns
        -------
        dict
            Outputs dictionary with keys:
            - stdout: list of stdout strings
            - stderr: list of stderr strings
            - errors: list of error dicts
            - display_data: list of display data dicts

        Raises
        ------
        TimeoutError
            If execution exceeds timeout and recovery fails
        """
        outputs = {
            'stdout': [],
            'stderr': [],
            'errors': [],
            'display_data': []
        }

        start_time = time.time()
        timeout_attempt = 0
        MAX_TIMEOUT_RETRIES = 2
        channel_attempt = 0
        MAX_CHANNEL_RECOVERIES = 2

        def _restart_after_channel_failure(reason: Exception) -> bool:
            nonlocal kc, outputs, start_time, channel_attempt
            if not auto_recover or channel_attempt >= MAX_CHANNEL_RECOVERIES:
                return False
            print(
                f"⚠ [CHANNEL] Kernel channel failure: {type(reason).__name__}: {reason}. "
                f"Attempting recovery ({channel_attempt + 1}/{MAX_CHANNEL_RECOVERIES})..."
            )
            if not self._recover_from_kernel_failure():
                return False
            channel_attempt += 1
            kc = self.current_session['kernel_client']
            outputs = {
                'stdout': [],
                'stderr': [],
                'errors': [],
                'display_data': []
            }
            start_time = time.time()
            return True

        try:
            msg_id = kc.execute(code, silent=False)
        except Exception as e:
            if _restart_after_channel_failure(e):
                msg_id = kc.execute(code, silent=False)
            else:
                raise

        while True:
            if time.time() - start_time > self.timeout:
                if auto_recover and timeout_attempt < MAX_TIMEOUT_RETRIES:
                    # Attempt recovery: interrupt -> restart
                    print(f"⏱ [TIMEOUT] Timeout after {self.timeout}s, attempting recovery "
                          f"(attempt {timeout_attempt + 1}/{MAX_TIMEOUT_RETRIES})...")

                    if self._recover_from_kernel_failure():
                        # Recovery succeeded, retry execution
                        timeout_attempt += 1
                        kc = self.current_session['kernel_client']
                        start_time = time.time()
                        msg_id = kc.execute(code, silent=False)
                        # Clear previous outputs
                        outputs = {
                            'stdout': [],
                            'stderr': [],
                            'errors': [],
                            'display_data': []
                        }
                        print(f"⚙ [RESTART] Retrying execution after recovery...")
                        continue
                    else:
                        # Recovery failed
                        raise TimeoutError(
                            f"Execution exceeded {self.timeout}s timeout and recovery failed "
                            f"after {timeout_attempt + 1} attempts"
                        )

                # No auto-recovery or max retries exceeded
                raise TimeoutError(
                    f"Execution exceeded {self.timeout}s timeout"
                    + (f" (after {timeout_attempt} recovery attempts)" if timeout_attempt > 0 else "")
                )

            try:
                msg = kc.get_iopub_msg(timeout=1.0)
            except Empty:
                continue
            except Exception as e:
                if _restart_after_channel_failure(e):
                    msg_id = kc.execute(code, silent=False)
                    print("⚙ [RESTART] Retrying execution after channel recovery...")
                    continue
                raise

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg['msg_type']
            content = msg['content']

            if msg_type == 'stream':
                if content['name'] == 'stdout':
                    outputs['stdout'].append(content['text'])
                elif content['name'] == 'stderr':
                    outputs['stderr'].append(content['text'])

            elif msg_type == 'error':
                outputs['errors'].append({
                    'ename': content['ename'],
                    'evalue': content['evalue'],
                    'traceback': content['traceback']
                })

            elif msg_type == 'display_data':
                outputs['display_data'].append(content['data'])

            elif msg_type == 'execute_result':
                outputs['display_data'].append(content['data'])

            elif msg_type == 'status' and content['execution_state'] == 'idle':
                break

        return outputs

    def _execute_in_session(self, code: str, adata) -> Any:
        """
        Execute code in current session, maintaining context.

        This method handles the high-level execution flow:
        - Saves adata to temp file
        - Injects adata into kernel namespace
        - Executes user code
        - Extracts result adata back

        Parameters
        ----------
        code : str
            Python code to execute
        adata : AnnData
            Input AnnData object

        Returns
        -------
        AnnData
            Result AnnData object after execution

        Raises
        ------
        RuntimeError
            If execution fails
        """
        kc = self.current_session['kernel_client']
        session_dir = self.current_session['session_dir']

        # Save adata to temp file
        temp_input = session_dir / f"temp_input_{self.session_prompt_count}.h5ad"
        adata.write_h5ad(temp_input)

        # Inject adata into kernel namespace
        inject_code = f"""
import anndata
adata = anndata.read_h5ad('{temp_input}')
print(f"✓ Loaded: {{adata.shape[0]}} cells × {{adata.shape[1]}} genes")
"""
        inject_outputs = self._execute_code_in_kernel(inject_code, kc)
        if inject_outputs['stdout']:
            print(''.join(inject_outputs['stdout']))

        # Execute user's generated code
        print(f"⚙ Executing prompt {self.session_prompt_count + 1}...")
        outputs = self._execute_code_in_kernel(code, kc)

        # Display outputs
        if outputs['stdout']:
            print(''.join(outputs['stdout']))

        if outputs['errors']:
            error = outputs['errors'][0]
            traceback_str = '\n'.join(error['traceback'])
            raise RuntimeError(
                f"Execution failed: {error['ename']}: {error['evalue']}\n"
                f"{traceback_str}\n\n"
                f"See session notebook: {self.current_session['notebook_path']}"
            )

        # Extract adata back from kernel
        temp_output = session_dir / f"temp_output_{self.session_prompt_count}.h5ad"
        extract_code = f"""
adata.write_h5ad('{temp_output}')
print(f"✓ Saved: {{adata.shape[0]}} cells × {{adata.shape[1]}} genes")
"""
        extract_outputs = self._execute_code_in_kernel(extract_code, kc)
        if extract_outputs['stdout']:
            print(''.join(extract_outputs['stdout']))

        # Load result
        try:
            import anndata
            result_adata = anndata.read_h5ad(temp_output)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load result adata: {e}\n"
                f"Execution may have succeeded but adata object was not properly saved."
            )

        # Append to session notebook
        self._append_to_session_notebook(code, outputs)

        return result_adata

    def execute(self, code: str, adata, execution_id: Optional[str] = None):
        """
        Execute code in session notebook.

        Manages session lifecycle:
        - Creates new session if none exists
        - Reuses session for prompts 2-5
        - Restarts session after max prompts

        Parameters
        ----------
        code : str
            Python code to execute
        adata : AnnData
            Input AnnData object
        execution_id : Optional[str]
            Unique identifier for this execution

        Returns
        -------
        AnnData
            Result AnnData object after execution

        Raises
        ------
        RuntimeError
            If code execution fails

        Examples
        --------
        >>> executor = SessionNotebookExecutor()
        >>> result = executor.execute("adata = adata[adata.obs['n_genes'] > 200]", adata)
        """
        # Check if need new session
        if self._should_start_new_session():
            self._start_new_session()

        # Execute in current session
        try:
            result_adata = self._execute_in_session(code, adata)
            self.session_prompt_count += 1

            # Log execution
            self.current_session['executions'].append({
                'execution_id': execution_id or f"exec_{self.session_prompt_count}",
                'timestamp': datetime.now().isoformat(),
                'prompt_number': self.session_prompt_count,
                'success': True
            })

            # Show session status to user
            print(f"⚙ = Session: Prompt {self.session_prompt_count}/{self.max_prompts_per_session}")
            if self.session_prompt_count >= self.max_prompts_per_session:
                print(f"💡 Next prompt will start a new session")

            return result_adata

        except Exception as e:
            # Log error
            if self.current_session:
                self.current_session['executions'].append({
                    'execution_id': execution_id or f"exec_{self.session_prompt_count + 1}",
                    'timestamp': datetime.now().isoformat(),
                    'prompt_number': self.session_prompt_count + 1,
                    'success': False,
                    'error': str(e)
                })
            raise

    # ===================================================================
    # Read-only execution (no adata round-trip)
    # ===================================================================

    def execute_readonly(self, code: str, adata) -> dict:
        """Execute code in the kernel session for read-only inspection.

        Unlike :meth:`execute`, this method does **not** serialize *adata*
        back from the kernel after execution.  It only injects *adata* into
        the kernel (if not already present in this session) and returns the
        captured stdout / stderr / errors.

        This avoids the expensive ``write_h5ad`` + ``read_h5ad`` round-trip
        that can crash kernels on large datasets (e.g. 170 K features).

        Returns
        -------
        dict
            ``{"stdout": str, "stderr": str, "error": str | None}``
        """
        # Ensure a live session exists
        if self._should_start_new_session():
            self._start_new_session()

        kc = self.current_session['kernel_client']
        session_dir = self.current_session['session_dir']

        # Inject adata into the kernel if this is the first prompt in the
        # session (prompt_count == 0 means nothing has been sent yet).
        if self.session_prompt_count == 0:
            temp_input = session_dir / "temp_input_readonly.h5ad"
            adata.write_h5ad(temp_input)
            inject_code = (
                "import anndata\n"
                f"adata = anndata.read_h5ad('{temp_input}')\n"
                "print(f'✓ Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes')\n"
            )
            inject_out = self._execute_code_in_kernel(inject_code, kc)
            if inject_out['stdout']:
                print(''.join(inject_out['stdout']))

        # Run the user's snippet
        outputs = self._execute_code_in_kernel(code, kc)

        stdout = ''.join(outputs.get('stdout', []))
        stderr = ''.join(outputs.get('stderr', []))
        error = None
        if outputs.get('errors'):
            err = outputs['errors'][0]
            error = f"{err['ename']}: {err['evalue']}"

        # Record in notebook (lightweight, no adata)
        self._append_to_session_notebook(code, outputs)

        return {"stdout": stdout, "stderr": stderr, "error": error}

    # ===================================================================
    # Cleanup
    # ===================================================================

    def shutdown(self):
        """Gracefully shutdown current session."""
        if self.current_session:
            self._archive_current_session()
            self.current_session = None

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except:
            pass


# ===================================================================
# Helper Function for Kernel Setup
# ===================================================================

def setup_kernel_for_env(conda_env: Optional[str] = None, auto_install: bool = False) -> bool:
    """
    Setup Jupyter kernel for current or specified conda environment.

    This is a user-facing helper function that checks if a kernel exists for
    the conda environment and optionally installs it.

    Parameters
    ----------
    conda_env : str, optional
        Conda environment name. If None, uses current environment.
    auto_install : bool, default False
        If True, automatically install kernel without prompting.
        If False, prompt user for confirmation.

    Returns
    -------
    bool
        True if kernel is installed/available, False otherwise.

    Raises
    ------
    RuntimeError
        If kernel installation fails.

    Examples
    --------
    >>> import omicverse as ov
    >>> # Check and setup kernel for current environment
    >>> ov.setup_kernel_for_env()
    >>> # Auto-install kernel without prompting
    >>> ov.setup_kernel_for_env(auto_install=True)
    >>> # Setup for specific environment
    >>> ov.setup_kernel_for_env(conda_env='myenv', auto_install=True)
    """
    import subprocess

    try:
        from jupyter_client.kernelspec import KernelSpecManager
    except ImportError:
        raise RuntimeError(
            "jupyter_client not installed.\n"
            "Install with: pip install jupyter-client ipykernel\n"
            "Or: conda install jupyter ipykernel"
        )

    # Detect conda environment
    if conda_env is None:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if not conda_env:
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix:
                conda_env = Path(conda_prefix).name
            else:
                conda_env = 'base'

    print(f"⚙ Setting up Jupyter kernel for conda environment: '{conda_env}'")

    # Check if kernel exists
    ksm = KernelSpecManager()
    available_kernels = ksm.find_kernel_specs()

    possible_names = [
        f'conda-env-{conda_env}-py',
        conda_env,
        'python3'
    ]

    for name in possible_names:
        if name in available_kernels:
            print(f"✓ [OK] Kernel '{name}' already installed and available")
            return True

    # Kernel not found
    print(f"⚠ [WARN] No kernel found for environment '{conda_env}'")
    print(f"\nAvailable kernels:")
    for kernel_name in sorted(available_kernels.keys()):
        print(f"  - {kernel_name}")

    # Prompt or auto-install
    if not auto_install:
        print(f"\n💡 [TIP] To use OmicVerse Agent, you need to install a kernel.")
        response = input("Install kernel now? [Y/n]: ").strip().lower()
        if response not in ['', 'y', 'yes']:
            print(f"\n⚠ [WARN] Skipping installation. You can install it later with:")
            print(f"   python -m ipykernel install --user --name {conda_env}")
            return False

    # Install kernel
    print(f"\n⚙ Installing kernel for '{conda_env}'...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'ipykernel', 'install',
            '--user', '--name', conda_env,
            '--display-name', f'Python ({conda_env})'
        ], capture_output=True, text=True, check=True)

        print(f"✓ [OK] Kernel installed successfully!")
        print(f"   Kernel name: {conda_env}")
        print(f"   Display name: Python ({conda_env})")

        # Verify installation
        ksm = KernelSpecManager()
        available_kernels = ksm.find_kernel_specs()
        if conda_env in available_kernels or f'conda-env-{conda_env}-py' in available_kernels:
            print(f"✓ [OK] Verified: Kernel is now available")
            return True
        else:
            print(f"⚠ [WARN] Warning: Kernel installed but not found in kernel list")
            return False

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to install kernel for '{conda_env}'\n"
            f"Error: {e.stderr}\n\n"
            f"Make sure ipykernel is installed:\n"
            f"  conda install -n {conda_env} ipykernel"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            f"ipykernel not found. Install it first:\n"
            f"  conda install ipykernel"
        )
