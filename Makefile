
.PHONY: all check download process zip clean coords plot setup viz cli \
        install uninstall rain rain6 rain-train rain-predict rain-now eval-plots \
        hourly

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
	@LAT="$(LAT)" LON="$(LON)" bash scripts/fetch_weather.sh
	@python3 scripts/process_weather.py
	@zip -j results/results.zip results/summary.txt

plot:
	@python3 scripts/plot_weather.py

viz: all plot
	@echo "📊 Charts generated."

setup:
	@python3 -m venv .venv
	@. .venv/bin/activate && pip install -r requirements.txt

cli:
	@python3 scripts/weather_cli.py --city Lagos --lat 6.5244 --lon 3.3792

install:
	@. .venv/bin/activate && pip install -e .

uninstall:
	@. .venv/bin/activate && pip uninstall -y weather-data-fetcher

rain:
	@python3 scripts/train_classify_rain.py

rain6:
	@python3 scripts/train_classify_rain_hourly.py

rain-train:
	@python3 scripts/train_rain_dual_thresholds.py

rain-predict:
	@python3 scripts/predict_rain.py

rain-now:
	@weather-cli rain --mode recall

eval-plots:
	@python3 scripts/plot_pr_roc.py

hourly:
	@LAT="$(LAT)" LON="$(LON)" PAST_DAYS="$(PAST_DAYS)" bash scripts/fetch_weather.sh
	@python3 scripts/export_hourly.py

.PHONY: xgb-train
xgb-train:
	@python3 scripts/train_xgb_12h.py

.PHONY: xgb-train-cal
xgb-train-cal:
	@python3 scripts/train_xgb_12h_calibrated.py
