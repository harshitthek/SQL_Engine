help:
	@echo 'Text-to-SQL Pipeline'

data:
	python run_all.py --task all
prompt:
	python scripts/run_prompts.py
