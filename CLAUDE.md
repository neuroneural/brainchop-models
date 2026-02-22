# Execution Policy

Claude may automatically execute the following commands without asking:

- npm install
- npm run build
- npm test
- git status
- git diff
- git add
- git commit (never push)
- docker build
- make

Claude must always ask before:

- git push
- rm -rf
- database migrations
- terraform apply
- anything touching production
