#!/bin/bash
if [ -d "outputs" ]; then
    rm -rf outputs
fi

for example in examples/*.py; do
    if [ -f "$example" ]; then
        echo "Running example: $example"
        uv run "$example"
        if [ $? -ne 0 ]; then
            echo "Error running $example"
            exit 1
        fi
    fi
done
