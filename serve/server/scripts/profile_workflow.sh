#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# GPU Profiling Script
#
# This script runs nvidia-smi to continuously capture GPU metrics to a CSV file.
# It automatically submits an inference request, monitors its status, and stops
# when the workflow completes. It then summarizes the data and outputs a JSON summary.
#
# Usage:
#   profile_workflow.sh --wf_name <workflow> --wf_json <json_file> --server_url <url> --ep_token <token>
#
# Arguments:
#   --wf_name     - Name of the workflow to execute (required)
#   --wf_json     - Path to JSON file containing inference request (required)
#   --server_url  - Inference server URL (default: http://localhost:8000)
#   --ep_token    - Authentication token (required)
#
# Note: Since models stay resident in memory, make sure each profiling run starts with
# a brand new entry point that has just been restarted. This prevents prior run models
# from polluting the memory footprint and skewing the results.
#

set -e

# Initialize variables
WORKFLOW_NAME=""
INFER_JSON=""
SERVER_URL="http://localhost:8000"
EP_TOKEN=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --wf_name)
            WORKFLOW_NAME="$2"
            shift 2
            ;;
        --wf_json)
            INFER_JSON="$2"
            shift 2
            ;;
        --server_url)
            SERVER_URL="$2"
            shift 2
            ;;
        --ep_token)
            EP_TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --wf_name <workflow> --wf_json <json_file> --server_url <url> --ep_token <token>"
            echo ""
            echo "Arguments:"
            echo "  --wf_name     - Name of the workflow to execute (required)"
            echo "  --wf_json     - Path to JSON file containing inference request (required)"
            echo "  --server_url  - Inference server URL (default: http://localhost:8000)"
            echo "  --ep_token    - Authentication token (required)"
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$WORKFLOW_NAME" ]; then
    echo "Error: --wf_name is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$INFER_JSON" ]; then
    echo "Error: --wf_json is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$EP_TOKEN" ]; then
    echo "Error: --ep_token is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate JSON file exists
if [ ! -f "$INFER_JSON" ]; then
    echo "Error: JSON file not found: $INFER_JSON"
    exit 1
fi

echo "==================================================================="
echo "GPU PROFILING WITH WORKFLOW EXECUTION"
echo "==================================================================="
echo ""
echo "Workflow: $WORKFLOW_NAME"
echo "JSON file: $INFER_JSON"
echo "Server: $SERVER_URL"
echo ""

# Generate random 8-character base64 salt
SALT=$(head -c 6 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9' | head -c 8)
echo "Generated salt: $SALT"
echo ""

# Output files with salt
OUTPUT_FILE="profile_${SALT}.csv"
OUTPUT_SUMMARY="outputs_${SALT}.txt"

# PID of nvidia-smi process
NVIDIA_SMI_PID=""

# Function to stop nvidia-smi
stop_nvidia_smi() {
    echo ""
    echo "Stopping nvidia-smi data collection..."

    # Kill nvidia-smi if it's still running
    if [ -n "$NVIDIA_SMI_PID" ]; then
        kill $NVIDIA_SMI_PID 2>/dev/null || true
        wait $NVIDIA_SMI_PID 2>/dev/null || true
    fi
}

# Function to process and summarize the profile data
summarize_and_output() {
    # Check if output file exists and has data
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "Error: No profile data collected"
        exit 1
    fi

    # Check if file has more than just the header
    line_count=$(wc -l < "$OUTPUT_FILE")
    if [ "$line_count" -le 1 ]; then
        echo "Error: No data rows collected (only header present)"
        exit 1
    fi

    echo ""
    echo "==================================================================="
    echo "GPU PROFILING SUMMARY"
    echo "==================================================================="
    echo ""

    # Process the CSV file
    # Read header line
    header=$(head -n 1 "$OUTPUT_FILE")

    # Convert header to array (split by comma)
    IFS=',' read -ra columns <<< "$header"

    # Trim whitespace from column names
    for i in "${!columns[@]}"; do
        columns[$i]=$(echo "${columns[$i]}" | xargs)
    done

    # Initialize arrays for max values and averages
    declare -A max_values
    declare -A max_numeric_values
    declare -A sum_values
    declare -A count_values

    # Initialize with very small values for max, and zero for sums
    for col in "${columns[@]}"; do
        max_numeric_values["$col"]=-999999999
        max_values["$col"]=""
        sum_values["$col"]=0
        count_values["$col"]=0
    done

    # Process each data row (skip header)
    row_count=0

    # Read data lines into array first
    mapfile -t data_lines < <(tail -n +2 "$OUTPUT_FILE")

    # Temporarily disable exit-on-error for data processing
    set +e

    # Use index-based for loop to process all data rows
    for ((idx=0; idx<${#data_lines[@]}; idx++)); do
        ((row_count++))
        line="${data_lines[$idx]}"

        # Skip empty lines
        if [ -z "$line" ]; then
            continue
        fi

        # Split line into array
        IFS=',' read -ra values <<< "$line"

        for i in "${!values[@]}"; do
            if [ $i -lt ${#columns[@]} ]; then
                col="${columns[$i]}"
                val=$(echo "${values[$i]}" | xargs)  # Trim whitespace

                # Try to extract numeric value (remove units like MiB, %, etc.)
                numeric_val=$(echo "$val" | grep -oE '[0-9]+\.?[0-9]*' | head -1)

                if [ -n "$numeric_val" ]; then
                    # Compare numeric values for max
                    current_max="${max_numeric_values[$col]}"

                    # Use awk for comparison
                    comparison=$(awk -v n="$numeric_val" -v m="$current_max" 'BEGIN { print (n > m) ? 1 : 0 }')

                    if [ "$comparison" = "1" ]; then
                        max_numeric_values["$col"]=$numeric_val
                        max_values["$col"]=$val
                    fi

                    # Accumulate sum for average calculation (for utilization metrics)
                    if [[ "$col" == *"utilization"* ]]; then
                        current_sum="${sum_values[$col]}"
                        sum_values["$col"]=$(awk -v s="$current_sum" -v n="$numeric_val" 'BEGIN { print s + n }')
                        count_values["$col"]=$((count_values["$col"] + 1))
                    fi
                else
                    # For non-numeric values, just keep the first non-empty one
                    if [ -z "${max_values[$col]}" ] && [ -n "$val" ]; then
                        max_values["$col"]=$val
                    fi
                fi
            fi
        done
    done

    # Re-enable exit-on-error
    set -e

    # Compute averages for utilization metrics
    declare -A avg_values
    for col in "${columns[@]}"; do
        if [[ "$col" == *"utilization"* ]] && [ "${count_values[$col]}" -gt 0 ]; then
            avg_values["$col"]=$(awk -v s="${sum_values[$col]}" -v c="${count_values[$col]}" 'BEGIN { printf "%.2f", s / c }')
        fi
    done

    # Clear/create the output file
    > "$OUTPUT_SUMMARY"

    # Print summary for all columns (to both stdout and file)
    {
        echo ""
        echo "Average values for utilization metrics:"
        echo "-----------------------------------------------------------------"
        for col in "${columns[@]}"; do
            if [[ "$col" == *"utilization"* ]]; then
                # Extract unit from the max value
                unit=""
                if [[ "${max_values[$col]}" == *"%"* ]]; then
                    unit=" %"
                fi
                echo "$col: ${avg_values[$col]}${unit}"
            fi
        done

        echo ""
        echo "Peak values for each metric:"
        echo "-----------------------------------------------------------------"
        for col in "${columns[@]}"; do
            echo "$col: ${max_values[$col]}"
        done
    } | tee -a "$OUTPUT_SUMMARY"

    # Generate JSON for specific columns
    # Extract the specific metrics we need by searching for column name patterns
    gpu_util_peak=""
    mem_util_peak=""
    gpu_util_avg=""
    mem_util_avg=""
    mem_used=""
    mem_total=""

    for col in "${columns[@]}"; do
        if [[ "$col" == *"utilization.gpu"* ]]; then
            gpu_util_peak="${max_values[$col]}"
            gpu_util_avg="${avg_values[$col]}"
            if [ -n "$gpu_util_avg" ]; then
                gpu_util_avg="${gpu_util_avg} %"
            fi
        fi
        if [[ "$col" == *"utilization.memory"* ]]; then
            mem_util_peak="${max_values[$col]}"
            mem_util_avg="${avg_values[$col]}"
            if [ -n "$mem_util_avg" ]; then
                mem_util_avg="${mem_util_avg} %"
            fi
        fi
        if [[ "$col" == *"memory.used"* ]]; then
            mem_used="${max_values[$col]}"
        fi
        if [[ "$col" == *"memory.total"* ]]; then
            mem_total="${max_values[$col]}"
        fi
    done

    # Generate JSON with nested structure (to both stdout and file)
    {
        echo ""
        echo "==================================================================="
        echo "JSON SUMMARY (Key Metrics)"
        echo "==================================================================="
        echo ""
        echo "{"
        echo "  \"workflow\": \"${WORKFLOW_NAME}\","
        echo "  \"gpus.used\": 1,"
        echo "  \"average\": {"
        echo "    \"utilization.gpu\": \"${gpu_util_avg:-N/A}\","
        echo "    \"utilization.memory\": \"${mem_util_avg:-N/A}\""
        echo "  },"
        echo "  \"peak\": {"
        echo "    \"utilization.gpu\": \"${gpu_util_peak:-N/A}\","
        echo "    \"utilization.memory\": \"${mem_util_peak:-N/A}\","
        echo "    \"memory.used\": \"${mem_used:-N/A}\","
        echo "    \"memory.total\": \"${mem_total:-N/A}\""
        echo "  }"
        echo "}"
        echo ""
    } | tee -a "$OUTPUT_SUMMARY"

    echo "Summary and JSON output also saved to: $OUTPUT_SUMMARY"

    exit 0
}

# Remove old profile file if it exists
rm -f "$OUTPUT_FILE"

# Start nvidia-smi in the background
echo "Starting GPU profiling..."
nvidia-smi --query-gpu=timestamp,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv --loop=1 -f "$OUTPUT_FILE" &

# Save the PID
NVIDIA_SMI_PID=$!

# Give nvidia-smi a moment to start
sleep 2

# Submit inference request
echo "Submitting inference request..."

RESPONSE=$(curl -k -s -w "\n%{http_code}" --max-time 30 -X POST "${SERVER_URL}/v1/infer/${WORKFLOW_NAME}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${EP_TOKEN}" \
    -d @${INFER_JSON} 2>&1)
CURL_EXIT=$?

echo "Curl finished with exit code: $CURL_EXIT"
echo "Raw response:"
echo "$RESPONSE"
echo "---"

# Extract HTTP status code (last line)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
# Extract response body (everything except last line)
RESPONSE_BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP Code: $HTTP_CODE"
echo "Response Body: $RESPONSE_BODY"

# Check HTTP status
if [ "$HTTP_CODE" != "200" ]; then
    echo "Error: Inference submission failed with HTTP status $HTTP_CODE"
    echo "Response: $RESPONSE_BODY"
    stop_nvidia_smi
    exit 1
fi

# Parse execution_id from response
EXEC_ID=$(echo "$RESPONSE_BODY" | grep -o '"execution_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$EXEC_ID" ]; then
    echo "Error: Could not extract execution_id from response"
    echo "Response: $RESPONSE_BODY"
    stop_nvidia_smi
    exit 1
fi

# Check initial status
INITIAL_STATUS=$(echo "$RESPONSE_BODY" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

if [ "$INITIAL_STATUS" != "queued" ]; then
    echo "Error: Expected status 'queued' but got '$INITIAL_STATUS'"
    echo "Response: $RESPONSE_BODY"
    stop_nvidia_smi
    exit 1
fi

echo "Inference submitted successfully"
echo "Execution ID: $EXEC_ID"
echo "Initial status: $INITIAL_STATUS"
echo ""
echo "Monitoring workflow execution..."

# Poll for status until completed
while true; do
    sleep 5

    # Get status
    STATUS_RESPONSE=$(curl -k -s -w "\n%{http_code}" --max-time 30 -X GET \
        "${SERVER_URL}/v1/infer/${WORKFLOW_NAME}/${EXEC_ID}/status" \
        -H "accept: application/json" \
        -H "Authorization: Bearer ${EP_TOKEN}" 2>&1)

    # Extract HTTP status code
    STATUS_HTTP_CODE=$(echo "$STATUS_RESPONSE" | tail -n1)
    # Extract response body
    STATUS_BODY=$(echo "$STATUS_RESPONSE" | sed '$d')

    # Check HTTP status
    if [ "$STATUS_HTTP_CODE" != "200" ]; then
        echo "Error: Status check failed with HTTP status $STATUS_HTTP_CODE"
        echo "Response: $STATUS_BODY"
        stop_nvidia_smi
        exit 1
    fi

    # Parse status
    EXEC_STATUS=$(echo "$STATUS_BODY" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$EXEC_STATUS" ]; then
        echo "Error: Could not extract status from response"
        echo "Response: $STATUS_BODY"
        stop_nvidia_smi
        exit 1
    fi

    echo "  Status: $EXEC_STATUS"

    # Check if failed
    if [ "$EXEC_STATUS" = "failed" ]; then
        echo "Error: Workflow execution failed"
        echo "Response: $STATUS_BODY"
        stop_nvidia_smi
        exit 1
    fi

    # Check if completed
    if [ "$EXEC_STATUS" = "completed" ]; then
        echo ""
        echo "Workflow execution completed successfully!"
        break
    fi
done

# Stop nvidia-smi
stop_nvidia_smi

# Rename files from salt to execution ID
NEW_OUTPUT_FILE="profile_${EXEC_ID}.csv"
NEW_OUTPUT_SUMMARY="outputs_${EXEC_ID}.txt"

echo ""
echo "Renaming files to use execution ID..."
mv "$OUTPUT_FILE" "$NEW_OUTPUT_FILE"
OUTPUT_FILE="$NEW_OUTPUT_FILE"
OUTPUT_SUMMARY="$NEW_OUTPUT_SUMMARY"

echo "Profile file: $OUTPUT_FILE"
echo "Summary file: $OUTPUT_SUMMARY"
echo ""

# Process and output summary
summarize_and_output
