@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

for /l %%m in (4,1,20) do (
for /l %%x in (6, 1, 6) do (
	set pow=1
	for /l %%y in (1,1,%%x) do (
		SET /A pow*=10
	)
	gen_network.exe !pow! %%m 100
)
)

