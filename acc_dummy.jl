@everywhere push!(LOAD_PATH, pwd()); 
@everywhere LOAD_PATH=unique(LOAD_PATH);
@everywhere include("accelerated_alg.jl"); 
run_test()

