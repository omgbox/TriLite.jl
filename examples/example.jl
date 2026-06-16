"""
TriLite.jl - Example Usage

This script demonstrates how to use the TriLite inference engine.
"""
module Example

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using TriLite

function main()
    println("BitNet.jl Example")
    println("=" ^ 50)

    # Example 1: Load model (uncomment when you have a GGUF file)
    # model = load_model("path/to/bitnet-2B.gguf")

    # Example 2: Generate text
    # response = generate(model, "What is the capital of France?")
    # println("Response: $response")

    # Example 3: Interactive chat
    # chat(model)

    # Example 4: Custom generation parameters
    # response = generate(model, "Tell me a joke",
    #                     max_tokens=100,
    #                     temperature=0.7f0,
    #                     top_k=40,
    #                     top_p=0.9f0)

    println("\nTo use this example:")
    println("1. Download a BitNet GGUF model")
    println("2. Uncomment the load_model line above")
    println("3. Run: julia example.jl")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module Example
