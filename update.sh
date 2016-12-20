#!/bin/bash

# create directories

mkdir -p lectures
mkdir -p assignments
mkdir -p assignments/assignment1
mkdir -p assignments/assignment2
mkdir -p assignments/assignment3

# copy lecture slides

cp "../lectures/lecture1/slides/slides.pdf" "lectures/lecture1.pdf"
cp "../lectures/lecture2/slides/slides.pdf" "lectures/lecture2.pdf"
cp "../lectures/lecture3/slides/slides.pdf" "lectures/lecture3.pdf"
cp "../lectures/lecture4/slides/slides.pdf" "lectures/lecture4.pdf"
cp "../lectures/lecture5/slides/slides.pdf" "lectures/lecture5.pdf"
cp "../lectures/lecture6/slides/slides.pdf" "lectures/lecture6.pdf"
cp "../lectures/lecture7/slides/slides.pdf" "lectures/lecture7.pdf"
cp "../lectures/lecture8/slides/slides.pdf" "lectures/lecture8.pdf"
cp "../lectures/lecture9/slides/slides.pdf" "lectures/lecture9.pdf"

# copy general assignment information

cp "../assignments/general.md" "assignments/"
cp "../assignments/server.md" "assignments/"
cp "../assignments/groups.md" "assignments/"

# copy assignment 1 information

cp "../assignments/assignment1/specification/part1.md" "assignments/assignment1/"
cp "../assignments/assignment1/specification/part2.md" "assignments/assignment1/"
cp "../assignments/assignment1/specification/part3.md" "assignments/assignment1/"
cp "../assignments/assignment1/submissions/recap/slides.pdf" "assignments/assignment1/recap.pdf"

# copy assignment 2 information

cp "../assignments/assignment2/specification/part1.md" "assignments/assignment2/"
cp "../assignments/assignment2/specification/part2.md" "assignments/assignment2/"
cp "../assignments/assignment2/specification/part3.md" "assignments/assignment2/"
cp "../assignments/assignment2/specification/part4.md" "assignments/assignment2/"
cp "../assignments/assignment2/specification/cat.jpg" "assignments/assignment2/"

# copy assignment 3 information

cp "../assignments/assignment3/specification/spec.md" "assignments/assignment3/"
cp "../assignments/assignment3/leaderboard.md" "assignments/assignment3/"
