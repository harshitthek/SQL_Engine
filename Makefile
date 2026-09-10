.PHONY: help install install-train data test test-deployment test-all \
       smoke-test eval serve gradio lint format clean docker-build docker-up

help:
	@echo "Text-to-SQL Engine — Makefile"
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install deployment dependencies"
	@echo "    make install-train    Install all dependencies (deployment + training)"
	@echo ""
	@echo "  Data:"
	@echo "    make data             Run data preparation pipeline (Tasks 02-09)"
	@echo ""
	@echo "  Test:"
	@echo "    make test             Run core pipeline + evaluator tests"
	@echo "    make test-deployment  Run deployment tests (API, Gradio, DB)"
	@echo "    make test-all         Run all tests"
	@echo "    make smoke-test       Fast 2-step training smoke test"
	@echo ""
	@echo "  Serve:"
	@echo "    make serve            Start FastAPI inference API (port 8000)"
	@echo "    make gradio           Start Gradio UI (port 7860)"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make eval             Run batch evaluation on Spider dev set"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint             Run ruff linter"
	@echo "    make format           Format code with ruff"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build     Build Docker image"
	@echo "    make docker-up        Start services with docker compose"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean            Remove caches and build artifacts"

install:
	pip install -r requirements.txt

install-train:
	pip install -r requirements-train.txt

data:
	python run_all.py --task all

test:
	pytest tests/ -v

test-deployment:
	cd deployment && pytest -v

test-all:
	pytest tests/ -v
	cd deployment && pytest -v

smoke-test:
	python scripts/smoke_test_pipeline.py

eval:
	python scripts/run_eval.py

serve:
	cd deployment && python -m uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

gradio:
	cd deployment && python app.py

lint:
	ruff check .

format:
	ruff format . || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

docker-build:
	docker build -t sql-engine .

docker-up:
	docker compose up --build
