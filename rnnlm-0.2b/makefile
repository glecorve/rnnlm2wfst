SRC=src/

all:
	@(cd $(SRC) && $(MAKE) $@)
 
clean:
	@(cd $(SRC) && $(MAKE) $@)

mrproper: clean
	@(rm bin/{rnnlm,rnn2fst,wfst-ppl,compute-mapping,trace-hidden-layer})

