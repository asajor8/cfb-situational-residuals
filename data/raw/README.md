# Raw CFBD downloads (not committed)

`src/download.py` writes cached JSON here (play-by-play, FBS team lists, and SP+ ratings).

Raw play-by-play is excluded from the repository because of size and because it should be pulled with your own [CollegeFootballData](https://collegefootballdata.com) API key. Set `CFBD_API_KEY` (see `.env.example`) and run the pipeline to populate this folder.
