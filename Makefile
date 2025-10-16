.PHONY: all check download process plot zip clean coords

all: check download process zip
	@echo "🏁 Weather pipeline complete."

check:
	@command -v curl >/dev/null || (echo "curl missing"; exit 1)
	@command -v python3 >/dev/null || (echo "python3 missing"; exit 1)

download:
	@bash scripts/fetch_weather.sh

process:
	@python3 scripts/process_weather.py

plot: process
	@python3 scripts/plot_weather.py

zip:
	@zip -j results/results.zip results/summary.txt

clean:
	@rm -rf data results logs
	@mkdir -p data results logs

# Usage: make coords LAT=5.6037 LON=-0.1870
coords: check
	@bash scripts/fetch_weather.sh $(LAT) $(LON)
	@python3 scripts/process_weather.py
	@zip -j results/results.zip results/summary.txt
	@echo "🌍 Coords run complete."
