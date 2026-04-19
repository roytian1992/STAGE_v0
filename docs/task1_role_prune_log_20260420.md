# Task1 Role Prune Log 2026-04-20

## Scope

- Repo: `STAGE_v0`
- Release path: `/vepfs-mlp2/c20250513/241404044/users/roytian/STAGE_v0`
- Task touched in this cleanup commit: `task1` only
- Files intentionally excluded from this commit: `task2`, `task3`, tools, runs, and other dirty local artifacts

## Rule

Delete any focal role whose `timeline_node_count` is less than or equal to half of the second-highest focal-role node count in the same movie.

## Counts

- Movies scanned: `151`
- Task1 focal roles before cleanup: `447`
- Task1 focal roles after cleanup: `435`
- Roles removed: `12`

## Removed Roles

- `Chinese/ch031fd9f8e8339ed61190f6e72d73dc8b05e8aef2`: `小四父亲`
- `Chinese/ch0c733d3c1e5b335c7a44a0c18d42e0b8ec40a841`: `宫羽田`
- `Chinese/ch264d4181722e2f969e974d0f57ae7dea83d4bef4`: `王老师`
- `Chinese/ch3a57b2f5c710ef650faa2af0210b76d702c456a7`: `张队长`
- `Chinese/chad83aa5c1d76c9068cb7f2693704827931d91b36`: `林焕清`
- `Chinese/chb23b5d80b64a100b58872d47658af25d2191f4cd`: `姜茂才`
- `English/en225b7ddc26e6473e8954b591202296c3`: `LaLoosh`
- `English/en7f8648f5449c4f9da46febfd68cdbd58`: `Pauly`
- `English/en8dc9b5e23f324c86a3593ba801f672ac`: `Marilyn`
- `English/enb844ecabcc7f4bc380cc87b5d3e24348`: `Liz Sherman`
- `English/enec3061f1b9d346fa84285a8f3eb111ba`: `Jack`
- `English/enf70932f1126d4ce791057d9a8b377ada`: `Lucas`

## Notes

- `task1 -> task3` normalized role coverage was audited separately and the true missing-role count is `0`; this cleanup is about weak third-role pruning, not cross-task recovery.
- A previous local batch also modified some `task2` and `task3` files, but those files are intentionally not included in this commit.
