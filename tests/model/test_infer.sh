#!/bin/bash

python scripts/model/reddit/infer.py \
	models/reddit/US_TextSubredditTime/model.joblib \
	tests/model/test_infer_users.txt \
	tests/model/test_infer_output.csv \
	--start_date 2018-01-01 \
	--end_date 2020-01-01 \
	--comment_limit 100 \
	--known_coordinates \
	--reverse_geocode
