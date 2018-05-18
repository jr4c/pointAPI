
install:
	pip install -r requirements.txt
.PHONY: install

#server will need to be running
test:
	curl -F file=@var/data/sample-0.png http://localhost:8585/point/classify

debug:
	python server.py

var/data/mnist.h5:
	python train_mnist.py
