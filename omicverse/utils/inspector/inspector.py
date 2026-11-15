"""
Main DataStateInspector class for runtime prerequisite validation.

This module provides the primary interface for validating prerequisites
before function execution in OmicVerse workflows.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from anndata import AnnData

from .validators import DataValidators
from .prerequisite_checker import PrerequisiteChecker
from .suggestion_engine import SuggestionEngine
from .llm_formatter import LLMFormatter, OutputFormat
from .data_structures import (
    ValidationResult,
    DataCheckResult,
    Suggestion,
)


class DataStateInspector:
    """Runtime validator for OmicVerse prerequisite tracking.

    This class inspects AnnData objects to validate that all prerequisites
    for a given function are satisfied before execution. It checks both
    data structure requirements and prerequisite function execution.

    Attributes:
        adata: The AnnData object to inspect.
        registry: Function registry with Layer 1 metadata.
        validators: DataValidators instance for checking data structures.
        prerequisite_checker: PrerequisiteChecker for detecting executed functions.
        suggestion_engine: SuggestionEngine for generating fix suggestions.
        llm_formatter: LLMFormatter for LLM-friendly output.

    Example:
        >>> from omicverse.utils.registry import get_registry
        >>> inspector = DataStateInspector(adata, get_registry())
        >>> result = inspector.validate_prerequisites('leiden')
        >>> if not result.is_valid:
        ...     print(result.message)
        ...     # Get LLM-formatted output
        ...     formatted = inspector.format_for_llm(result)
        ...     print(formatted)
    """

    def __init__(self, adata: AnnData, registry: Any):
        """Initialize inspector with AnnData and function registry.

        Args:
            adata: The AnnData object to inspect.
            registry: Function registry with Layer 1 metadata.
        """
        self.adata = adata
        self.registry = registry
        self.validators = DataValidators(adata)
        self.prerequisite_checker = PrerequisiteChecker(adata, registry)
        self.suggestion_engine = SuggestionEngine(registry)
        self.llm_formatter = LLMFormatter()

    def validate_prerequisites(self, function_name: str) -> ValidationResult:
        """Validate all prerequisites for a given function.

        This is the main entry point for validation. It checks:
        1. Required data structures exist
        2. Required prerequisite functions have been executed (Phase 2)
        3. Generates suggestions to fix any issues

        Args:
            function_name: Name of the function to validate.

        Returns:
            ValidationResult with comprehensive validation details.

        Example:
            >>> result = inspector.validate_prerequisites('leiden')
            >>> print(result.is_valid)
            >>> print(result.missing_data_structures)
        """
        # Get function metadata from registry
        func_meta = self._get_function_metadata(function_name)

        if not func_meta:
            return ValidationResult(
                function_name=function_name,
                is_valid=False,
                message=f"Function '{function_name}' not found in registry",
            )

        # Check data requirements
        data_check = self.check_data_requirements(function_name)

        # Phase 2: Check prerequisite functions
        prereq_results = self.prerequisite_checker.check_all_prerequisites(function_name)

        # Determine which prerequisite functions are missing
        missing_prereqs = []
        executed_funcs = []
        confidence_scores = {}

        for prereq_func, detection_result in prereq_results.items():
            confidence_scores[prereq_func] = detection_result.confidence
            if detection_result.executed:
                executed_funcs.append(prereq_func)
            else:
                missing_prereqs.append(prereq_func)

        # Check if all requirements are satisfied
        all_valid = data_check.is_valid and len(missing_prereqs) == 0

        if all_valid:
            return ValidationResult(
                function_name=function_name,
                is_valid=True,
                message=f"All requirements satisfied for {function_name}",
                data_check_result=data_check,
                executed_functions=executed_funcs,
                confidence_scores=confidence_scores,
            )

        # Generate comprehensive suggestions using SuggestionEngine (Phase 3)
        suggestions = self.suggestion_engine.generate_suggestions(
            function_name=function_name,
            missing_prerequisites=missing_prereqs,
            missing_data=data_check.all_missing_structures,
            data_check_result=data_check,
        )

        return ValidationResult(
            function_name=function_name,
            is_valid=False,
            message=f"Missing requirements for {function_name}",
            missing_prerequisites=missing_prereqs,
            missing_data_structures=data_check.all_missing_structures,
            executed_functions=executed_funcs,
            confidence_scores=confidence_scores,
            suggestions=suggestions,
            data_check_result=data_check,
        )

    def check_data_requirements(self, function_name: str) -> DataCheckResult:
        """Check if required data structures exist for a function.

        Args:
            function_name: Name of the function to check.

        Returns:
            DataCheckResult with details about missing/present structures.

        Example:
            >>> result = inspector.check_data_requirements('leiden')
            >>> print(result.all_missing_structures)
        """
        func_meta = self._get_function_metadata(function_name)

        if not func_meta:
            # Return invalid result if function not found
            return DataCheckResult(is_valid=False)

        requires = func_meta.get('requires', {})
        return self.validators.check_all_requirements(requires)

    def _get_function_metadata(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a function from the registry.

        Args:
            function_name: Name of the function.

        Returns:
            Metadata dict or None if not found.
        """
        try:
            # Access registry to get function metadata
            # The exact method depends on registry implementation
            if hasattr(self.registry, 'get_function'):
                return self.registry.get_function(function_name)
            elif hasattr(self.registry, 'functions'):
                return self.registry.functions.get(function_name)
            else:
                # Try direct dictionary access
                return self.registry.get(function_name)
        except Exception:
            return None

    def _generate_data_suggestions(
        self,
        function_name: str,
        data_check: DataCheckResult,
        func_meta: Dict[str, Any],
    ) -> List[Suggestion]:
        """Generate suggestions to fix missing data requirements.

        Args:
            function_name: Name of the target function.
            data_check: Result of data requirements check.
            func_meta: Function metadata from registry.

        Returns:
            List of Suggestion objects in priority order.
        """
        suggestions = []
        missing = data_check.all_missing_structures

        # Get prerequisite information
        prerequisites = func_meta.get('prerequisites', {})
        required_funcs = prerequisites.get('functions', [])
        auto_fix = func_meta.get('auto_fix', 'none')

        # Generate suggestions based on missing structures
        if 'obsp' in missing:
            # Missing graph structures typically means neighbors not run
            if 'connectivities' in missing['obsp'] or 'distances' in missing['obsp']:
                suggestions.append(Suggestion(
                    priority='HIGH',
                    suggestion_type='direct_fix',
                    description='Compute neighbor graph (required for clustering/UMAP)',
                    code='ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)',
                    explanation=(
                        'The function requires a neighbor graph (connectivities and distances). '
                        'Run ov.pp.neighbors() to compute the k-nearest neighbors graph.'
                    ),
                    estimated_time='10-30 seconds',
                    estimated_time_seconds=20,
                    prerequisites=['pca'] if 'X_pca' not in self.adata.obsm else [],
                    impact='Enables clustering and graph-based analyses',
                    auto_executable=(auto_fix == 'auto'),
                ))

        if 'obsm' in missing:
            # Missing embeddings
            if 'X_pca' in missing['obsm']:
                suggestions.append(Suggestion(
                    priority='HIGH',
                    suggestion_type='direct_fix',
                    description='Run PCA for dimensionality reduction',
                    code='ov.pp.scale(adata)\nov.pp.pca(adata, n_pcs=50)',
                    explanation=(
                        'PCA is required but not found. Scale the data and run PCA.'
                    ),
                    estimated_time='5-20 seconds',
                    estimated_time_seconds=10,
                    prerequisites=['preprocess', 'scale'],
                    impact='Required for downstream analyses',
                    auto_executable=False,
                ))

            if 'X_umap' in missing['obsm']:
                suggestions.append(Suggestion(
                    priority='MEDIUM',
                    suggestion_type='direct_fix',
                    description='Compute UMAP embedding',
                    code='ov.pp.umap(adata)',
                    explanation=(
                        'UMAP embedding is required. Ensure neighbors graph exists first.'
                    ),
                    estimated_time='10-60 seconds',
                    estimated_time_seconds=30,
                    prerequisites=['neighbors'],
                    impact='Required for visualization',
                    auto_executable=False,
                ))

        if 'obs' in missing:
            # Missing observation columns
            obs_missing = missing['obs']
            if any('leiden' in col or 'louvain' in col for col in obs_missing):
                suggestions.append(Suggestion(
                    priority='HIGH',
                    suggestion_type='direct_fix',
                    description='Run clustering to generate cluster labels',
                    code='ov.pp.leiden(adata, resolution=1.0)',
                    explanation=(
                        'Cluster labels are required. Run Leiden or Louvain clustering.'
                    ),
                    estimated_time='5-30 seconds',
                    estimated_time_seconds=15,
                    prerequisites=['neighbors'],
                    impact='Generates cell cluster assignments',
                    auto_executable=False,
                ))

        # If we have required functions listed, suggest running them
        if required_funcs and auto_fix == 'escalate':
            func_list = ', '.join(required_funcs)
            suggestions.append(Suggestion(
                priority='HIGH',
                suggestion_type='workflow_guidance',
                description=f'Complete prerequisite workflow: {func_list}',
                code=f'# Run prerequisite functions: {func_list}\n# See function documentation for details',
                explanation=(
                    f'{function_name} requires a complex workflow. '
                    f'Ensure these functions have been run: {func_list}'
                ),
                estimated_time='Varies',
                prerequisites=required_funcs,
                impact='Satisfies all prerequisites',
                auto_executable=False,
            ))

        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 999))

        return suggestions

    def _generate_prerequisite_suggestions(
        self,
        function_name: str,
        missing_prereqs: List[str],
        func_meta: Dict[str, Any],
    ) -> List[Suggestion]:
        """Generate suggestions for missing prerequisite functions.

        Args:
            function_name: Name of the target function.
            missing_prereqs: List of missing prerequisite function names.
            func_meta: Function metadata from registry.

        Returns:
            List of Suggestion objects in priority order.
        """
        suggestions = []

        if not missing_prereqs:
            return suggestions

        # Map common functions to their typical code
        function_code_map = {
            'qc': 'ov.pp.qc(adata, tresh={"mito_perc": 20, "n_genes_by_counts": 2500})',
            'preprocess': 'ov.pp.preprocess(adata, mode="shiftlog|pearson", n_HVGs=2000)',
            'scale': 'ov.pp.scale(adata)',
            'pca': 'ov.pp.pca(adata, n_pcs=50)',
            'neighbors': 'ov.pp.neighbors(adata, n_neighbors=15, n_pcs=50)',
            'umap': 'ov.pp.umap(adata)',
            'leiden': 'ov.pp.leiden(adata, resolution=1.0)',
            'louvain': 'ov.pp.louvain(adata, resolution=1.0)',
        }

        # Generate suggestions for each missing prerequisite
        for prereq in missing_prereqs:
            code = function_code_map.get(prereq, f'# Run {prereq} function\n# See documentation for details')

            # Determine priority based on how critical the prerequisite is
            priority = 'CRITICAL' if prereq in ['qc', 'preprocess'] else 'HIGH'

            suggestions.append(Suggestion(
                priority=priority,
                suggestion_type='prerequisite',
                description=f'Run prerequisite function: {prereq}',
                code=code,
                explanation=(
                    f'{function_name} requires {prereq} to be executed first. '
                    f'This function prepares the data for downstream analysis.'
                ),
                estimated_time='10-60 seconds',
                estimated_time_seconds=30,
                prerequisites=[],  # No nested prerequisites for now
                impact=f'Satisfies prerequisite: {prereq}',
                auto_executable=False,
            ))

        # If multiple prerequisites missing, add a workflow suggestion
        if len(missing_prereqs) > 1:
            prereq_chain = ' -> '.join(missing_prereqs)
            suggestions.append(Suggestion(
                priority='HIGH',
                suggestion_type='workflow',
                description=f'Complete prerequisite workflow chain',
                code=f'# Execute in order: {prereq_chain}\n' + '\n'.join(
                    function_code_map.get(p, f'# {p}()') for p in missing_prereqs
                ),
                explanation=(
                    f'{function_name} requires multiple prerequisite functions. '
                    f'Execute them in order: {prereq_chain}'
                ),
                estimated_time='1-5 minutes',
                estimated_time_seconds=180,
                prerequisites=[],
                impact='Completes full prerequisite chain',
                auto_executable=False,
            ))

        return suggestions

    def get_validation_summary(self, function_name: str) -> Dict[str, Any]:
        """Get a dictionary summary of validation suitable for LLM consumption.

        Args:
            function_name: Name of the function to validate.

        Returns:
            Dict with validation details in LLM-friendly format.

        Example:
            >>> summary = inspector.get_validation_summary('leiden')
            >>> print(summary['missing_data_structures'])
        """
        result = self.validate_prerequisites(function_name)
        return result.get_summary()

    def format_for_llm(
        self,
        result: ValidationResult,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
    ) -> str:
        """Format validation result for LLM consumption.

        Args:
            result: ValidationResult to format.
            output_format: Desired output format (markdown, plain_text, json, prompt).

        Returns:
            Formatted string in the specified format.

        Example:
            >>> result = inspector.validate_prerequisites('leiden')
            >>> formatted = inspector.format_for_llm(result, OutputFormat.MARKDOWN)
            >>> print(formatted)
        """
        return self.llm_formatter.format_validation_result(result, output_format)

    def get_llm_prompt(
        self,
        function_name: str,
        task: str = "Fix the validation errors",
    ) -> str:
        """Get LLM prompt for validation issues.

        Args:
            function_name: Name of the function to validate.
            task: Task description for the LLM.

        Returns:
            LLMPrompt with system and user prompts.

        Example:
            >>> prompt = inspector.get_llm_prompt('leiden', "Fix preprocessing issues")
            >>> print(prompt)
        """
        result = self.validate_prerequisites(function_name)
        return self.llm_formatter.create_agent_prompt(result, task)

    def get_natural_language_explanation(self, function_name: str) -> str:
        """Get natural language explanation of validation result.

        Args:
            function_name: Name of the function to validate.

        Returns:
            Natural language explanation suitable for users.

        Example:
            >>> explanation = inspector.get_natural_language_explanation('leiden')
            >>> print(explanation)
        """
        result = self.validate_prerequisites(function_name)
        return self.llm_formatter.format_natural_language(result)
