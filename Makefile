help:
	@echo 'Text-to-SQL Pipeline'

data:
	python run_all.py --task all
prompt:
	python scripts/run_prompts.py
schema:
	python scripts/run_schema.py
study:
	python scripts/run_sample_study.py
tokens:
	python scripts/run_token_dist.py
