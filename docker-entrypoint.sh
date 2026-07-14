#!/bin/sh
set -e

# Ensure the models directory and its subdirectories exist and are writable
# by appuser. This is needed because PaaS providers like Coolify may mount
# persistent volumes under /app/models that are owned by root.
mkdir -p /app/models/zoning /app/models/yield /app/models/data /app/models/models
chown -R appuser:appgroup /app/models 2>/dev/null || true
chmod -R 775 /app/models 2>/dev/null || true

# Ensure the Hugging Face cache directory is writable by appuser.
# By default HF writes to ~/.cache/huggingface which may not exist or be
# writable when running as a non-root user inside a minimal container.
mkdir -p /app/.cache/huggingface
chown -R appuser:appgroup /app/.cache 2>/dev/null || true
chmod -R 775 /app/.cache 2>/dev/null || true

# Run database migrations before starting the application.
su -s /bin/sh appuser -c 'alembic upgrade head'

# Drop privileges and execute the main command as appuser
exec su -s /bin/sh appuser -c 'exec "$@"' sh "$@"

