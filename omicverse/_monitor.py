import sys
import time
import textwrap
import anndata
import numpy as np
import pandas as pd
import functools
from datetime import datetime
from scipy import sparse

# Global flag to control monitor output display
_SHOW_MONITOR_OUTPUT = True
# Global nesting level counter
_MONITOR_NESTING_LEVEL = 0

def set_monitor_display(show: bool):
    """Set whether to display monitor output.
    
    Args:
        show: If True, display monitor output; if False, suppress it.
    """
    global _SHOW_MONITOR_OUTPUT
    _SHOW_MONITOR_OUTPUT = show

def get_monitor_display() -> bool:
    """Get the current monitor display setting.
    
    Returns:
        True if monitor output is enabled, False otherwise.
    """
    return _SHOW_MONITOR_OUTPUT

# ANSI escape codes for styling
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RED = "\033[31m"

class StructureWatcher:
    def __init__(self, adata, func_name="Unknown Function"):
        self.adata = adata
        self.func_name = func_name
        # Capture start state
        self.start_time = time.time()
        self.start_state = self._snapshot(adata)

    def _get_type_desc(self, obj):
        """Helper to get a clean description of an object's type and shape."""
        shape_str = ""
        if hasattr(obj, "shape"):
            shape_str = f"{obj.shape[0]}x{obj.shape[1]}" if len(obj.shape) == 2 else str(obj.shape)
        
        if sparse.issparse(obj):
            return f"(sparse matrix, {shape_str})"
        elif isinstance(obj, np.ndarray):
            return f"(array, {shape_str})"
        elif isinstance(obj, pd.DataFrame):
            return f"(dataframe, {shape_str})"
        elif isinstance(obj, dict):
            return "(dictionary)"
        else:
            return f"({type(obj).__name__})"

    def _get_column_type(self, series):
        """Helper to get a simplified type description for a Series."""
        dtype = series.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            return "category"
        dt_str = str(dtype)
        if "int" in dt_str: return "int"
        if "float" in dt_str: return "float"
        if "object" in dt_str or "str" in dt_str: return "str"
        return dt_str

    def _snapshot(self, adata):
        """Captures keys, types, and shapes for comparison."""
        obs_types = {}
        for c in adata.obs.columns:
            try:
                obs_types[c] = self._get_column_type(adata.obs[c])
            except Exception:
                obs_types[c] = "unknown (error)"

        var_types = {}
        for c in adata.var.columns:
            try:
                var_types[c] = self._get_column_type(adata.var[c])
            except Exception:
                var_types[c] = "unknown (error)"        

        snapshot = {
            "shape": adata.shape,
            "obs": obs_types,
            "var": var_types,
            "uns": set(adata.uns.keys()),
            # For complex slots, store a dict of {key: description_string}
            "obsm": {k: self._get_type_desc(v) for k, v in adata.obsm.items()},
            "varm": {k: self._get_type_desc(v) for k, v in adata.varm.items()},
            "layers": {k: self._get_type_desc(v) for k, v in adata.layers.items()},
            "obsp": {k: self._get_type_desc(v) for k, v in adata.obsp.items()},
        }
        return snapshot

    def finish(self, error=None):
        """Compares states and prints the formatted report.

        If ``error`` is not None the wrapped function raised before
        returning; render a FAILED summary that embeds the exception
        (type + message) instead of a success-styled box, so the failure
        is unmistakable rather than masquerading as a 0.0s no-op.
        """
        end_time = time.time()
        duration = round(end_time - self.start_time, 4)

        if error is not None:
            self._print_failure(duration, error)
            return

        end_state = self._snapshot(self.adata)
        self._print_report(self.start_state, end_state, duration)
        self._log_to_uns(self.start_state, end_state, duration)

    # --- failure rendering -------------------------------------------------
    def _emit_row(self, colored, visible, width, border):
        """Print one box row, padding to ``width`` using the visible length."""
        V = f"{border}│{Style.RESET}"
        padding = " " * max(0, width - 2 - len(visible))
        print(f"{V}{colored}{padding}{V}")

    def _print_failure(self, duration, error):
        """Render a red FAILED box that embeds the raised exception."""
        if not _SHOW_MONITOR_OUTPUT:
            return
        # Only the top-level (outermost) call prints, matching success output.
        if _MONITOR_NESTING_LEVEL > 0:
            return

        TL, H, TR = "╭", "─", "╮"
        BL, BR = "╰", "╯"
        width = 70
        title = f" SUMMARY: {self.func_name} [FAILED] "
        header_line = f"{TL}{H}{title}{H * (width - 3 - len(title))}{TR}"
        footer_line = f"{BL}{H * (width - 2)}{BR}"

        print(f"\n{Style.RED}{header_line}{Style.RESET}")

        # Duration
        self._emit_row(
            f"  Duration: {Style.YELLOW}{duration}s{Style.RESET}",
            f"  Duration: {duration}s", width, Style.RED,
        )
        # Status: which exception type was raised
        etype = type(error).__name__
        self._emit_row(
            f"  Status:   {Style.BOLD}{Style.RED}✖ raised {etype}{Style.RESET}",
            f"  Status:   ✖ raised {etype}", width, Style.RED,
        )
        # blank spacer
        print(f"{Style.RED}│{Style.RESET}" + " " * (width - 2) + f"{Style.RED}│{Style.RESET}")
        # ERROR header + rule
        self._emit_row(f"  {Style.BOLD}{Style.RED}ERROR{Style.RESET}", "  ERROR", width, Style.RED)
        self._emit_row(f"  {Style.DIM}─────{Style.RESET}", "  ─────", width, Style.RED)

        # Embedded, word-wrapped exception message (capped so the box
        # stays readable; full traceback still propagates to the caller).
        msg = str(error).strip() or "(no message)"
        lines = []
        for para in msg.splitlines() or [msg]:
            lines.extend(textwrap.wrap(para, width - 6) or [""])
        truncated = len(lines) > 8
        for line in lines[:8]:
            self._emit_row(
                f"  {Style.YELLOW}{line}{Style.RESET}", f"  {line}", width, Style.RED,
            )
        if truncated:
            self._emit_row(
                f"  {Style.DIM}… (truncated; see traceback above){Style.RESET}",
                "  … (truncated; see traceback above)", width, Style.RED,
            )

        print(f"{Style.RED}{footer_line}{Style.RESET}")

    def _print_report(self, start, end, duration):
        # Check if monitor output is enabled
        if not _SHOW_MONITOR_OUTPUT:
            return
        
        # Only print if we are at the top level (nesting level 0)
        if _MONITOR_NESTING_LEVEL > 0:
            return
        
        # Box drawing characters
        TL, H, TR = "╭", "─", "╮"
        V = "│"
        BL, BR = "╰", "╯"
        
        width = 70
        title = f" SUMMARY: {self.func_name} "
        title_len = len(title)
        header_line = f"{TL}{H}{title}{H * (width - 3 - title_len)}{TR}"
        footer_line = f"{BL}{H * (width - 2)}{BR}"
        
        print(f"\n{Style.BLUE}{header_line}{Style.RESET}")
        
        # Duration
        duration_text = f"  Duration: {Style.YELLOW}{duration}s{Style.RESET}"
        visible_len = len(f"  Duration: {duration}s")
        padding = " " * (width - 2 - visible_len)
        print(f"{Style.BLUE}{V}{Style.RESET}{duration_text}{padding}{Style.BLUE}{V}{Style.RESET}")
        
        # Shape
        # Use 'x' instead of '×' because '×' causes alignment issues in some terminals
        shape_str = f"{end['shape'][0]:,} x {end['shape'][1]:,}"
        if start['shape'] == end['shape']:
            shape_info = f"{Style.GREEN}{shape_str}{Style.RESET} {Style.DIM}(Unchanged){Style.RESET}"
            visible_shape = f"{shape_str} (Unchanged)"
        else:
            old_shape = f"{start['shape'][0]:,} x {start['shape'][1]:,}"
            shape_info = f"{Style.YELLOW}{old_shape}{Style.RESET} -> {Style.GREEN}{shape_str}{Style.RESET}"
            visible_shape = f"{old_shape} -> {shape_str}"
        
        shape_text = f"  Shape:    {shape_info}"
        visible_len = len(f"  Shape:    {visible_shape}")
        padding = " " * (width - 2 - visible_len)
        print(f"{Style.BLUE}{V}{Style.RESET}{shape_text}{padding}{Style.BLUE}{V}{Style.RESET}")
        
        print(f"{Style.BLUE}{V}{Style.RESET}" + " " * (width - 2) + f"{Style.BLUE}{V}{Style.RESET}")
        
        # Changes Header
        changes_text = f"  {Style.BOLD}{Style.CYAN}CHANGES DETECTED{Style.RESET}"
        visible_len = len("  CHANGES DETECTED")
        padding = " " * (width - 2 - visible_len)
        print(f"{Style.BLUE}{V}{Style.RESET}{changes_text}{padding}{Style.BLUE}{V}{Style.RESET}")
        
        separator_text = f"  {Style.DIM}────────────────{Style.RESET}"
        visible_len = len("  ────────────────")
        padding = " " * (width - 2 - visible_len)
        print(f"{Style.BLUE}{V}{Style.RESET}{separator_text}{padding}{Style.BLUE}{V}{Style.RESET}")

        # Sections - use 6-char width for consistency
        self._print_section("OBS", set(start['obs'].keys()), set(end['obs'].keys()), width, descriptions=end['obs'], title_width=6)
        self._print_section("VAR", set(start['var'].keys()), set(end['var'].keys()), width, descriptions=end['var'], title_width=6)
        self._print_section("UNS", start['uns'], end['uns'], width, is_uns=True, title_width=6)
        self._print_section("OBSP", set(start['obsp'].keys()), set(end['obsp'].keys()), width, descriptions=end['obsp'], title_width=6)
        self._print_section("OBSM", set(start['obsm'].keys()), set(end['obsm'].keys()), width, descriptions=end['obsm'], title_width=6)
        self._print_section("LAYERS", set(start['layers'].keys()), set(end['layers'].keys()), width, descriptions=end['layers'], title_width=6)

        print(f"{Style.BLUE}{footer_line}{Style.RESET}")

    def _print_section(self, title, start_keys, end_keys, width, is_uns=False, descriptions=None, title_width=6):
        added = sorted(list(end_keys - start_keys))
        V = f"{Style.BLUE}│{Style.RESET}"
        
        # Format title with consistent width
        title_formatted = f"{title:<{title_width}}"
        spaces_needed = 2 + title_width  # "● " + title + space before │
        
        if not added:
            return

        for i, key in enumerate(added):
            if i == 0:
                prefix_colored = f"  {Style.CYAN}●{Style.RESET} {Style.BOLD}{title_formatted}{Style.RESET} │"
                prefix_visible = f"  ● {title_formatted} │"
            else:
                # Match the spacing: "  ● " (4 chars) + title_width + " │" (2 chars)
                prefix_colored = f"  {' ' * spaces_needed} │"
                prefix_visible = f"  {' ' * spaces_needed} │"
            
            if is_uns:
                content = f"{prefix_colored} {Style.GREEN}✚{Style.RESET} {Style.YELLOW}{key}{Style.RESET}"
                visible_content = f"{prefix_visible} ✚ {key}"
                
                padding = " " * (width - 2 - len(visible_content))
                print(f"{V}{content}{padding}{V}")
                
                # Check for params
                if key in self.adata.uns:
                    val = self.adata.uns[key]
                    if isinstance(val, dict) and 'params' in val:
                        params_str = str(val['params'])
                        # Truncate if too long
                        max_param_len = width - 20
                        if len(params_str) > max_param_len:
                            params_str = params_str[:max_param_len-3] + "..."
                        
                        # Sub-line should align with the content, not the prefix
                        spaces_for_subline = 2 + spaces_needed + 1  # match prefix + space after │
                        sub_line = f"  {' ' * spaces_needed} │ {Style.DIM}└─ params:{Style.RESET} {Style.YELLOW}{params_str}{Style.RESET}"
                        visible_sub = f"  {' ' * spaces_needed} │ └─ params: {params_str}"
                        padding = " " * (width - 2 - len(visible_sub))
                        print(f"{V}{sub_line}{padding}{V}")
            else:
                desc = descriptions[key] if descriptions else ""
                desc = desc.strip("()")
                
                content = f"{prefix_colored} {Style.GREEN}✚{Style.RESET} {Style.YELLOW}{key}{Style.RESET} {Style.DIM}({desc}){Style.RESET}"
                visible_content = f"{prefix_visible} ✚ {key} ({desc})"
                
                # Handle potential overflow
                if len(visible_content) > width - 2:
                    max_len = width - 5
                    visible_content = visible_content[:max_len] + "..."
                    # Simplify colored version too
                    key_truncated = key[:max_len-20] + "..." if len(key) > max_len-20 else key
                    content = f"{prefix_colored} {Style.GREEN}✚{Style.RESET} {Style.YELLOW}{key_truncated}{Style.RESET} {Style.DIM}...{Style.RESET}"
                
                padding = " " * (width - 2 - len(visible_content))
                print(f"{V}{content}{padding}{V}")
        
        print(f"{V}" + " " * (width - 2) + f"{V}")

    def _log_to_uns(self, start, end, duration):
        """Save a short summary to adata.uns for permanence."""
        current_log = []
        if 'history_log' in self.adata.uns:
            log_entry = self.adata.uns['history_log']
            if isinstance(log_entry, pd.DataFrame):
                current_log = log_entry.to_dict('records')
            elif isinstance(log_entry, list):
                current_log = log_entry
            elif isinstance(log_entry, np.ndarray):
                # If loaded as recarray or object array
                if log_entry.dtype.names:
                    # It's a structured array
                    current_log = [dict(zip(log_entry.dtype.names, x)) for x in log_entry]
                else:
                    # Try to convert to list, assuming it might be array of dicts
                    current_log = list(log_entry)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "function": self.func_name,
            "duration": duration,
            "shape": str(end['shape'])
        }
        current_log.append(entry)
        self.adata.uns['history_log'] = pd.DataFrame(current_log)

# --- DECORATOR ---
def monitor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _MONITOR_NESTING_LEVEL
        _MONITOR_NESTING_LEVEL += 1
        
        # Auto-detect name if it's a partial or bound method
        name = getattr(func, "__qualname__", func.__name__)
        
        # Attempt to find 'adata' in arguments
        adata = None
        for arg in args:
            if isinstance(arg, anndata.AnnData):
                adata = arg
                break
        
        if adata is None:
            adata = kwargs.get("adata")

        watcher = None
        if adata is not None:
            watcher = StructureWatcher(adata, func_name=name)

        try:
            result = func(*args, **kwargs)
        finally:
            _MONITOR_NESTING_LEVEL -= 1
            if watcher:
                # sys.exc_info()[1] is the exception currently propagating
                # through this finally (None on the normal-return path).
                # The exception is NOT swallowed — it re-raises after finish.
                err = sys.exc_info()[1]
                try:
                    watcher.finish(error=err)
                except Exception:
                    # Monitoring must never mask the real error / result.
                    pass
        return result
    return wrapper
