#!/bin/bash

set -e

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Running migrations ==="
python manage.py migrate --noinput

echo "=== Collecting static files ==="
python manage.py collectstatic --noinput

echo "=== Creating superuser if not exists ==="
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@neurosense.ai', 'Aa245516@')
    print('Superuser created')
else:
    print('Superuser already exists')
" || echo "Superuser creation skipped"

echo "=== Starting gunicorn ==="
gunicorn --bind=0.0.0.0:8000 --timeout 600 --workers 2 Neurosense.wsgi:application