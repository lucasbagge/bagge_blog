---
title: Git revert
author: Lucas Bagge
date: '2021-07-13'
slug: []
categories:
  - gi
tags:
  - git revert
  - git
subtitle: ''
summary: ''
authors: []
lastmod: '2021-07-13T08:29:02+02:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

## Introduction
 
Often when building packages, application or models and using git as a service
tools the developer will encounter that a commit is creating problems. It can
also be that there has been an misunderstanding in the task and the commit
dosen´t solve the inital task.
 
One way to solve this issue is to use the commit command `git revert <git commit>`.
 
## Git revert
 
The use case is from a real practical example. I have created a solution on
a bug and commit the solutions. In the review of our sprint we discorver there
has been an misunderstanding and that there has not been a clear understanding
on what the commit should solve.
 
So we want to leave everything after that commit as it is but revert that
commit and solve the issue.
 
One example is to use `git revert`. It is a simple function that take
the commit id and revert that specific back. So it leave everythong else as
it was before but reverse the changes maded from that commit.
 
It would be like
 
```
git revert 123bd12
```
 
Here we have a commit `123bd12` and we have found out that it is wrong or
doing something that it should not do in the application. So we revert it and
can solve the task againg correctly.
 
## Conclusion
 
We have look at have we can solve a commit that was pushed to the developer
branch. Here we can use git revert to solve this issue.
