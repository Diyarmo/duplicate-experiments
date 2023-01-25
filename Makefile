COMMIT := $(shell git rev-parse HEAD 2> /dev/null || echo "NULL")
VERSION := $(shell git describe $(COMMIT) 2> /dev/null || echo "$(COMMIT)")
IMAGE_NAME := registry.cafebazaar.ir:5000/divar/review/bots/bots-dev/duplicate-bot
IMAGE_VERSION := $(VERSION)

UNIX_TIME := $(shell date +%s)

help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

docker: ## build docker image
ifdef CI_ACCESS_TOKEN
	docker build --build-arg no_proxy --build-arg http_proxy --build-arg https_proxy --build-arg ACCESS_TOKEN=$(CI_ACCESS_TOKEN) --build-arg GIT_SSH_PRIVATE_KEY=$(GIT_ACCESS_SSH_PRIVATE_KEY) -t $(IMAGE_NAME):$(IMAGE_VERSION) .
else
	@echo "\e[31mYou have to define CI_ACCESS_TOKEN enviroment variables. For more information read README.\e[0m"; exit 1
endif

push: docker ## push docker image to registry
ifdef CI_ACCESS_TOKEN
	docker push $(IMAGE_NAME):$(VERSION)
else
	@echo "You're trying to push an image \e[31mbuilt by your personal access token\e[0m."
	@echo "It's not a recommended practice. For more information read README."
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@docker push $(IMAGE_NAME):$(VERSION)
endif


tag-non-master : push docker
ifdef CI_ACCESS_TOKEN
	docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):test
	docker push $(IMAGE_NAME):test
else
	@echo "You're trying to tag non master branch an image \e[31mbuilt by your personal access token\e[0m."
	@echo "It's not a recommended practice. For more information read README."
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):test
endif

tag-master : push docker
ifdef CI_ACCESS_TOKEN
	docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest
	docker push $(IMAGE_NAME):latest
else
	@echo "You're trying to tag master branch an image \e[31mbuilt by your personal access token\e[0m."
	@echo "It's not a recommended practice. For more information read README."
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@docker tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest
endif



CI_ACCESS_TOKEN := ${CI_ACCESS_TOKEN}
GIT_ACCESS_SSH_PRIVATE_KEY := ${GIT_ACCESS_SSH_PRIVATE_KEY}

.PHONY: help docker push tag-non-master tag-master
