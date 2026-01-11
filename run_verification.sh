#!/bin/bash
# Master Verification Script
# Run this AFTER ingestion, BEFORE evaluation

set -e  # Exit on error

echo "======================================================================"
echo "AGENTIC GRAPHRAG VERIFICATION SUITE"
echo "======================================================================"
echo ""
echo "This will run 4 verification checks:"
echo "  1. System validation (validate_system.py)"
echo "  2. Execution path trace (trace_execution.py)"
echo "  3. Graph quality check (check_graph_quality.py)"
echo "  4. Fallback detection (check_fallbacks.py)"
echo ""
echo "Press ENTER to continue or Ctrl+C to cancel..."
read

echo ""
echo "======================================================================"
echo "CHECK 1/4: System Validation"
echo "======================================================================"
python validate_system.py
if [ $? -ne 0 ]; then
    echo "❌ System validation failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "CHECK 2/4: Execution Path Trace"
echo "======================================================================"
python trace_execution.py
if [ $? -ne 0 ]; then
    echo "⚠️  Execution trace had warnings (may be acceptable)"
fi

echo ""
echo "======================================================================"
echo "CHECK 3/4: Graph Quality Check"
echo "======================================================================"
python check_graph_quality.py
if [ $? -ne 0 ]; then
    echo "⚠️  Graph quality check had warnings (may be acceptable)"
fi

echo ""
echo "======================================================================"
echo "CHECK 4/4: Fallback Detection"
echo "======================================================================"
python check_fallbacks.py
if [ $? -ne 0 ]; then
    echo "⚠️  Fallback check had warnings (may be acceptable)"
fi

echo ""
echo "======================================================================"
echo "✅ VERIFICATION COMPLETE"
echo "======================================================================"
echo ""
echo "If all checks passed (or warnings are acceptable), you can now run:"
echo "  python evaluate_hotpotqa_full.py"
echo ""
