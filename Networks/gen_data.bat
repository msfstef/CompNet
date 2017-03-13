@echo off
SETLOCAL ENABLEDELAYEDEXPANSION


for /l %%L in (10,5,20) do (
for /l %%m in (1,1,3) do (
for /l %%x in (2, 1, 6) do (
	set pow=1
	for /l %%y in (1,1,%%x) do (
		SET /A pow*=10
	)
	gen_network.exe !pow! %%m 100 2 %%L
)
)
)



