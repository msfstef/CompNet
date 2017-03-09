@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

for /l %%m in (1,1,10) do (
for /l %%x in (2, 1, 7) do (
	set pow=1
	for /l %%y in (1,1,%%x) do (
		SET /A pow*=10
	)
	gen_network.exe !pow! %%m 1000 0
)
)


for /l %%m in (1,1,10) do (
for /l %%x in (2, 1, 7) do (
	set pow=1
	for /l %%y in (1,1,%%x) do (
		SET /A pow*=10
	)
	gen_network.exe !pow! %%m 1000 1
)
)

for /l %%L in (1,1,10) do (
for /l %%m in (1,1,10) do (
for /l %%x in (2, 1, 6) do (
	set pow=1
	for /l %%y in (1,1,%%x) do (
		SET /A pow*=10
	)
	gen_network.exe !pow! %%m 500 2 %%L
)
)
)



