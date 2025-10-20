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

.PHONY: cli
cli:
	@python3 scripts/weather_cli.py --city Lagos --lat 6.5244 --lon 3.3792

.PHONY: install uninstall
install:
	@. .venv/bin/activate && pip install -e .

uninstall:
	@. .venv/bin/activate && pip uninstall -y weather-data-fetcher

.PHONY: rain
rain:
	@python3 scripts/train_classify_rain.py

.PHONY: hourly rain6
hourly:
	@PAST_DAYS=14 bash scripts/fetch_weather.sh && python3 scripts/export_hourly.py

rain6:
	@python3 scripts/train_classify_rain_hourly.py

.PHONY: rain-train rain-predict
rain-train:
	@python3 scripts/train_rain_dual_thresholds.py

rain-predict:
	@python3 scripts/predict_rain.py

.PHONY: rain-now
rain-now:
	@python3 scripts/rain_cli.py --mode recall

.PHONY: eval-plots
eval-plots:
	@python3 scripts/plot_pr_roc.py

.PHONY: rain-train rain-now hourly
hourly:
	@PAST_DAYS=14 bash scripts/fetch_weather.sh && python3 scripts/export_hourly.py

rain-train:
	@python3 scripts/train_rain_dual_thresholds.py

rain-now:
	@python3 -c "import sys,subprocess; subprocess.run(['weather-cli','rain','--mode','recall'])"
