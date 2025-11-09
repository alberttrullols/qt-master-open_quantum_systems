#!/bin/bash

# Installation script for Open Quantum Systems Julia project
# This script sets up the Julia environment and installs required packages

echo "========================================="
echo "Open Quantum Systems - Setup Script"
echo "========================================="

# Check if Julia is installed
if ! command -v julia &> /dev/null; then
    echo "Error: Julia is not installed or not in PATH"
    echo "Please install Julia from https://julialang.org/downloads/"
    exit 1
fi

echo "Julia found: $(julia --version)"

# Check if we're in the project directory
if [ ! -f "Project.toml" ]; then
    echo "Error: Project.toml not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "Setting up Julia project environment..."

# Activate project environment and install packages
julia --project=. -e "
using Pkg
println(\"Instantiating project dependencies...\")
Pkg.instantiate()
println(\"✓ Dependencies installed successfully\")

println(\"Verifying installations...\")
try
    using QuantumOptics
    println(\"✓ QuantumOptics.jl loaded successfully\")
catch e
    println(\"✗ Error loading QuantumOptics.jl: \", e)
end

try  
    using Plots
    println(\"✓ Plots.jl loaded successfully\")
catch e
    println(\"✗ Error loading Plots.jl: \", e)
end

println(\"\\n=========================================\")
println(\"Setup complete!\")
println(\"=========================================\")
println(\"You can now run the simulations:\")
println(\"  julia dephasing.jl\")
println(\"  julia dephasing_rabi.jl\")
println(\"=========================================\")
"

echo "Setup completed successfully!"
