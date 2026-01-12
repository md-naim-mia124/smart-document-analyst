#!/bin/bash
# অব্যবহৃত ডকার ইমেজ, কন্টেইনার এবং ভলিউম ডিলিট করবে
docker system prune -a -f --volumes
# ১৫ দিনের বেশি পুরনো ব্যাকআপ ডিলিট করবে
find /home/ubuntu/db_backups -type f -mtime +15 -name "*.sql" -delete