./run_bench.sh -k nbodies -i rust-gpu,cuda-oxide -n 30 --version v9 -s "80000 500000 80000" && \
./run_bench.sh -k raytracer        -i rust-gpu,rust,cuda-oxide -n 30 --version v9 -s "10000 15000 20000" && \


------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------

./run_bench.sh -k ripple           -i rust-gpu,rust,cuda-oxide -n 30 --version v9 -s "16384 18432 20480" && \
./run_bench.sh -k nearest_neighbor -i rust-gpu,rust,cuda-oxide -n 30 --version v9 -s "100000000 200000000 300000000"