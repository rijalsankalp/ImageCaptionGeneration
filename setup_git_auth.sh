#!/usr/bin/env bash

# This script sets up GitHub authentication for pushing results from Azure ML

# Check if required environment variables are set
if [ -z "$GITHUB_USERNAME" ] || [ -z "$GITHUB_PAT" ]; then
    echo "Error: GITHUB_USERNAME and GITHUB_PAT environment variables must be set."
    echo "You can set these in the Azure ML workspace."
    exit 1
fi

# Set up Git credentials for HTTPS
git config --global credential.helper store
echo "https://$GITHUB_USERNAME:$GITHUB_PAT@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials

# Configure Git user
git config --global user.email "${GIT_EMAIL:-"$GITHUB_USERNAME@users.noreply.github.com"}"
git config --global user.name "${GIT_USERNAME:-"$GITHUB_USERNAME via Azure ML"}"

echo "Git credentials configured successfully."
echo "You can now use 'git push' commands in your training scripts."
