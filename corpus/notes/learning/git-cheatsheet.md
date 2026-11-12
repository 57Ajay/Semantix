# Git cheatsheet

- Undo the last commit but keep the changes: `git reset --soft HEAD~1`
- Find a lost commit: `git reflog`, then `git branch rescue <sha>`
- Find which commit broke something: `git bisect start`, `git bisect bad`,
  `git bisect good <sha>`, then test each step
- Rewrite the last few commits: `git rebase -i HEAD~3` (squash, reword, drop)
- Only one file from another branch: `git restore --source other-branch -- path/to/file`
- See who changed a line: `git blame -L 40,60 file.py`
- Stash including untracked files: `git stash -u`
