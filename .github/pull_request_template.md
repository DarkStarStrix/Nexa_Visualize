## Summary
- Describe the user-facing change in 2-4 bullets.

## Reliability Checklist
- [ ] Added/updated tests for new behavior.
- [ ] `npm run lint` passes locally.
- [ ] `npm test -- --watch=false --runInBand` passes locally.
- [ ] `npm run test:smoke` passes locally.
- [ ] `npm run build` passes locally.
- [ ] Session import/export remains backward compatible.
- [ ] No renderer/material/geometry leaks on mount/unmount paths.

## Risk Notes
- Main risk area:
- Rollback approach:
