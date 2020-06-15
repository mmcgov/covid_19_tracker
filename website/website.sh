#!/bin/sh
sudo systemctl daemon-reload
sudo systemctl stop covid_19_site
sudo systemctl stop nginx
sudo systemctl start covid_19_site
sudo systemctl start nginx
