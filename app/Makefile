build:
	RUSTFLAGS="-A warnings" cargo build

clean:
	rm -f *.cu *.h

run: build
	./target/debug/rust-metaprogramming

