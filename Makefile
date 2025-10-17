.PHONY: all check download process zip clean coords

all: check download process zip
	@echo "🏁 Weather pipeline complete."

check:
	@command -v curl >/dev/null || (echo "curl missing"; exit 1)
	@command -v python3 >/dev/null || (echo "python3 missing"; exit 1)
	@[ -f scripts/fetch_weather.sh ] || (echo "missing scripts/fetch_weather.sh"; exit 1)
	@[ -f scripts/process_weather.py ] || (echo "missing scripts/process_weather.py"; exit 1)

download:
	@bash scripts/fetch_weather.sh

process:
	@python3 scripts/process_weather.py

zip:
	@zip -j results/results.zip results/summary.txt results/summary.csv


clean:
	@rm -rf data results logs
	@mkdir -p data results logs

coords:
	@bash scripts/fetch_weather.sh $(LAT) $(LON)
	@python3 scripts/process_weather.py
	@zip -j results/results.zip results/summary.txt

.PHONY: plot
plot:
	@python3 scripts/plot_weather.py

.PHONY: setup
setup:
	@python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt

.PHONY: viz
viz: all plot
	@echo "📊 Charts generated."

