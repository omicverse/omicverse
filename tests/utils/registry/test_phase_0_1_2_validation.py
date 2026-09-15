"""
Source Code Validation Test for Phases 0, 1, and 2

Validates prerequisite metadata by inspecting source code directly,
avoiding the need for numpy and other dependencies.

Run with: python test_phase_0_1_2_validation.py
"""

import re
from pathlib import Path

print("=" * 80)
print("SOURCE CODE VALIDATION: PHASES 0, 1, AND 2")
print("=" * 80)

# ==============================================================================
# DEFINE ALL FUNCTIONS BY PHASE WITH FILE LOCATIONS
# ==============================================================================

phase_0_functions = [
    ('qc', 'omicverse/pp/_qc.py'),
    ('preprocess', 'omicverse/pp/_preprocess.py'),
    ('scale', 'omicverse/pp/_preprocess.py'),
    ('pca', 'omicverse/pp/_preprocess.py'),
    ('neighbors', 'omicverse/pp/_preprocess.py'),
]

phase_1_functions = [
    ('umap', 'omicverse/pp/_preprocess.py'),
    ('leiden', 'omicverse/pp/_preprocess.py'),
    ('score_genes_cell_cycle', 'omicverse/pp/_preprocess.py'),
    ('sude', 'omicverse/pp/_sude.py'),
    ('cosg', 'omicverse/single/_cosg.py'),
    ('TrajInfer', 'omicverse/single/_traj.py'),
]

phase_2_functions = [
    ('pySCSA', 'omicverse/single/_anno.py'),
    ('batch_correction', 'omicverse/single/_batch.py'),
    ('get_celltype_marker', 'omicverse/single/_anno.py'),
    ('scanpy_cellanno_from_dict', 'omicverse/single/_anno.py'),
    ('gptcelltype', 'omicverse/single/_gptcelltype.py'),
    ('get_cluster_celltype', 'omicverse/single/_cellvote.py'),
    ('DEG', 'omicverse/single/_deg_ct.py'),
    ('pyVIA', 'omicverse/single/_via.py'),
]

all_functions = phase_0_functions + phase_1_functions + phase_2_functions

print(f"\nTotal functions to validate: {len(all_functions)}")
print(f"  Phase 0 (Core preprocessing): {len(phase_0_functions)}")
print(f"  Phase 1 (Workflows): {len(phase_1_functions)}")
print(f"  Phase 2 (Annotation/Analysis): {len(phase_2_functions)}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def extract_decorator_block(file_path, function_name):
    """Extract the @register_function decorator block for a function."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Find the decorator block for this function
        # Look for @register_function followed by class/def function_name
        pattern = rf'@register_function\s*\((.*?)\)\s*(?:class|def)\s+{re.escape(function_name)}\s*[\(\[]'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return match.group(1)
        return None
    except Exception as e:
        return None

def check_parameter_exists(decorator_text, param_name):
    """Check if a parameter exists in the decorator."""
    if not decorator_text:
        return False
    return param_name in decorator_text

def extract_parameter_value(decorator_text, param_name):
    """Extract the value of a parameter from decorator text."""
    if not decorator_text:
        return None

    # Match parameter with dict or string value
    pattern = rf'{param_name}\s*=\s*(\{{.*?\}}|\'[^\']*\'|"[^"]*")'
    match = re.search(pattern, decorator_text, re.DOTALL)

    if match:
        return match.group(1)
    return None

# ==============================================================================
# TEST 1: Decorator Presence
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 1: @register_function Decorator Presence")
print("=" * 80)

decorator_found = []
decorator_missing = []

for func_name, file_path in all_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        print(f"✓ {func_name:30s} in {file_path}")
        decorator_found.append(func_name)
    else:
        print(f"✗ {func_name:30s} MISSING in {file_path}")
        decorator_missing.append(func_name)

print(f"\nResult: {len(decorator_found)}/{len(all_functions)} functions have @register_function decorator")

# ==============================================================================
# TEST 2: Required Parameters
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 2: Required Parameters (aliases, category, description)")
print("=" * 80)

required_params = ['aliases', 'category', 'description']
param_check = {param: [] for param in required_params}

for func_name, file_path in all_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        missing = []
        for param in required_params:
            if check_parameter_exists(decorator, param):
                param_check[param].append(func_name)
            else:
                missing.append(param)

        if missing:
            print(f"✗ {func_name:30s} missing: {', '.join(missing)}")
        else:
            print(f"✓ {func_name:30s} all required params present")

print("\nParameter coverage:")
for param in required_params:
    count = len(param_check[param])
    print(f"  {param:20s}: {count}/{len(all_functions)} functions")

# ==============================================================================
# TEST 3: New Prerequisite Metadata Parameters
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 3: New Prerequisite Metadata Parameters")
print("=" * 80)

prereq_params = ['prerequisites', 'requires', 'produces', 'auto_fix']
prereq_check = {param: [] for param in prereq_params}

for func_name, file_path in all_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        found = []
        missing = []
        for param in prereq_params:
            if check_parameter_exists(decorator, param):
                prereq_check[param].append(func_name)
                found.append(param)
            else:
                missing.append(param)

        status = "✓" if len(found) >= 3 else "⚠"  # At least 3 of 4 params
        print(f"{status} {func_name:30s} has: {', '.join(found) if found else 'NONE'}")

print("\nPrerequisite parameter coverage:")
for param in prereq_params:
    count = len(prereq_check[param])
    percentage = (count / len(all_functions)) * 100
    print(f"  {param:20s}: {count:2d}/{len(all_functions)} ({percentage:5.1f}%)")

# ==============================================================================
# TEST 4: Auto-fix Strategy Values
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 4: Auto-fix Strategy Values")
print("=" * 80)

valid_strategies = ['auto', 'escalate', 'none']
strategy_counts = {'auto': [], 'escalate': [], 'none': [], 'invalid': []}

for func_name, file_path in all_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator:
        auto_fix_value = extract_parameter_value(decorator, 'auto_fix')
        if auto_fix_value:
            # Remove quotes
            value = auto_fix_value.strip('\'"')
            if value in valid_strategies:
                strategy_counts[value].append(func_name)
                print(f"✓ {func_name:30s} auto_fix = '{value}'")
            else:
                strategy_counts['invalid'].append(func_name)
                print(f"✗ {func_name:30s} auto_fix = '{value}' (INVALID)")
        else:
            print(f"⚠ {func_name:30s} auto_fix not specified")

print("\nAuto-fix strategy distribution:")
for strategy, funcs in strategy_counts.items():
    if strategy != 'invalid':
        count = len(funcs)
        percentage = (count / len(all_functions)) * 100 if all_functions else 0
        print(f"  {strategy:10s}: {count:2d} functions ({percentage:5.1f}%)")

if strategy_counts['invalid']:
    print(f"\n⚠ Invalid strategies found in: {', '.join(strategy_counts['invalid'])}")

# ==============================================================================
# TEST 5: Prerequisites Parameter Structure
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 5: Prerequisites Parameter Structure")
print("=" * 80)

functions_with_prereqs = 0
functions_with_functions = 0
functions_with_optional = 0

for func_name, file_path in all_functions:
    full_path = Path(__file__).parent / file_path
    decorator = extract_decorator_block(str(full_path), func_name)

    if decorator and check_parameter_exists(decorator, 'prerequisites'):
        functions_with_prereqs += 1
        prereq_value = extract_parameter_value(decorator, 'prerequisites')

        has_functions = 'functions' in prereq_value if prereq_value else False
        has_optional = 'optional_functions' in prereq_value if prereq_value else False

        if has_functions:
            functions_with_functions += 1
        if has_optional:
            functions_with_optional += 1

        status = "✓" if (has_functions or has_optional) else "⚠"
        parts = []
        if has_functions:
            parts.append("required")
        if has_optional:
            parts.append("optional")

        print(f"{status} {func_name:30s} has {', '.join(parts) if parts else 'empty dict'}")

print(f"\nPrerequisites structure:")
print(f"  Functions with prerequisites parameter: {functions_with_prereqs}/{len(all_functions)}")
print(f"  Functions with 'functions' key: {functions_with_functions}")
print(f"  Functions with 'optional_functions' key: {functions_with_optional}")

# ==============================================================================
# TEST 6: Phase-by-Phase Summary
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 6: Phase-by-Phase Summary")
print("=" * 80)

def validate_phase(phase_name, functions):
    print(f"\n{phase_name}:")
    complete = 0
    partial = 0
    missing = 0

    for func_name, file_path in functions:
        full_path = Path(__file__).parent / file_path
        decorator = extract_decorator_block(str(full_path), func_name)

        if not decorator:
            status = "✗ MISSING"
            missing += 1
        else:
            # Check for all 4 new parameters
            has_count = sum([
                check_parameter_exists(decorator, 'prerequisites'),
                check_parameter_exists(decorator, 'requires'),
                check_parameter_exists(decorator, 'produces'),
                check_parameter_exists(decorator, 'auto_fix')
            ])

            if has_count == 4:
                status = "✓ COMPLETE"
                complete += 1
            elif has_count >= 2:
                status = f"⚠ PARTIAL ({has_count}/4)"
                partial += 1
            else:
                status = f"✗ MINIMAL ({has_count}/4)"
                missing += 1

        print(f"  {status:20s} {func_name}")

    total = len(functions)
    print(f"\n  Summary: {complete} complete, {partial} partial, {missing} missing (total: {total})")
    return complete, partial, missing

phase0_complete, phase0_partial, phase0_missing = validate_phase("Phase 0 (Core Preprocessing)", phase_0_functions)
phase1_complete, phase1_partial, phase1_missing = validate_phase("Phase 1 (Workflows)", phase_1_functions)
phase2_complete, phase2_partial, phase2_missing = validate_phase("Phase 2 (Annotation/Analysis)", phase_2_functions)

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

total_complete = phase0_complete + phase1_complete + phase2_complete
total_partial = phase0_partial + phase1_partial + phase2_partial
total_missing = phase0_missing + phase1_missing + phase2_missing

all_tests_passed = (
    len(decorator_found) == len(all_functions) and
    total_complete == len(all_functions) and
    len(strategy_counts['invalid']) == 0
)

print(f"""
{'✅' if len(decorator_found) == len(all_functions) else '✗'} TEST 1: Decorators - {len(decorator_found)}/{len(all_functions)} present
{'✅' if all(len(param_check[p]) == len(all_functions) for p in required_params) else '⚠'} TEST 2: Required Params - all functions covered
{'✅' if functions_with_prereqs >= len(all_functions) * 0.9 else '⚠'} TEST 3: Prerequisite Params - {functions_with_prereqs}/{len(all_functions)} have metadata
{'✅' if len(strategy_counts['invalid']) == 0 else '✗'} TEST 4: Auto-fix Values - all valid
{'✅' if functions_with_prereqs > 0 else '✗'} TEST 5: Prerequisites Structure - {functions_with_prereqs} functions
✅ TEST 6: Phase Validation - all phases reviewed

Overall Status:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 0 (Core): {phase0_complete}/{len(phase_0_functions)} complete, {phase0_partial} partial, {phase0_missing} missing
Phase 1 (Workflows): {phase1_complete}/{len(phase_1_functions)} complete, {phase1_partial} partial, {phase1_missing} missing
Phase 2 (Annotation): {phase2_complete}/{len(phase_2_functions)} complete, {phase2_partial} partial, {phase2_missing} missing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL: {total_complete}/{len(all_functions)} complete ({total_complete/len(all_functions)*100:.1f}%)
       {total_partial} partial, {total_missing} missing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implementation Quality: {'✅ EXCELLENT' if total_complete == len(all_functions) else '⚠ NEEDS REVIEW' if total_complete >= len(all_functions) * 0.8 else '✗ INCOMPLETE'}

Auto-fix Distribution:
- auto ({len(strategy_counts['auto'])}): {', '.join(strategy_counts['auto'][:3])}{'...' if len(strategy_counts['auto']) > 3 else ''}
- escalate ({len(strategy_counts['escalate'])}): {', '.join(strategy_counts['escalate'][:3])}{'...' if len(strategy_counts['escalate']) > 3 else ''}
- none ({len(strategy_counts['none'])}): {', '.join(strategy_counts['none'][:3])}{'...' if len(strategy_counts['none']) > 3 else ''}
""")

print("=" * 80)
print("Validation completed!")
print("=" * 80)
