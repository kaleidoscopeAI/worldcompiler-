# Goeckoh-Coupled Planetary Substrate + Entity Integration
CC      = gcc
CFLAGS  = -O3 -march=native -mavx2 -mfma -ffp-contract=fast -Wall -Wextra -Isrc
LDFLAGS = -lm
SUB     = src/substrate.c src/substrate_stream.c
BRIDGE  = src/entity_coherence.c

.PHONY: all test proof clean

all: substrate_test binding_test integration_test render_proof

substrate_test: $(SUB) src/test_substrate.c
	$(CC) $(CFLAGS) src/substrate.c src/test_substrate.c -o $@ $(LDFLAGS)

binding_test: $(SUB) src/test_binding.c
	$(CC) $(CFLAGS) $(SUB) src/test_binding.c -o $@ $(LDFLAGS)

integration_test: $(SUB) $(BRIDGE) src/test_integration.c
	$(CC) $(CFLAGS) $(SUB) $(BRIDGE) src/test_integration.c -o $@ $(LDFLAGS)

render_proof: src/substrate.c src/render_proof.c
	$(CC) $(CFLAGS) src/substrate.c src/render_proof.c -o $@ $(LDFLAGS)

test: substrate_test binding_test integration_test
	@echo "===== SUBSTRATE CORRECTNESS GATE =====" && ./substrate_test
	@echo "" && echo "===== GCL BINDING =====" && ./binding_test
	@echo "" && echo "===== FULL-STACK INTEGRATION =====" && ./integration_test

proof: render_proof
	@./render_proof
	@python3 -c "from PIL import Image; Image.open('proof.ppm').save('proof.png'); print('wrote proof.png')" 2>/dev/null || echo "proof.ppm written"

clean:
	rm -f substrate_test binding_test integration_test render_proof proof.ppm proof.png
