# Session: sesja_2025-10-22_23-10

**Created:** 2025-10-22_23-10 UTC

## Quick Reference
- **Job ID:** 1190904
- **Log Path:** ********************************************************* SSH server at Warsaw University of Technology Faculty of Mathematics and Information Science System.Management.Automation.RemoteException Individuals using this computer system without authority, or in  excess of  their  authority, are subject to having all  of  their activities  on this  system monitored  and recorded   by   system  personnel.  In   the   course  of monitoring  individuals improperly  using this system, or in the  course of system  maintenance, the  activities of authorized  users  may  also be  monitored.  Anyone using this system  expressly consents to such monitoring and is advised that if such monitoring reveals possible criminal activity , system personnel may  provide the  evidence of such monitoring to law enforcement officials. ********************************************************* /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_ddp_6gpu_dgx2_500ep_1190904.log
- **Description:** DETR 6 GPU training on dgx-2 - resume from epoch 140
- **Folder:** sesja_2025-10-22_23-10

## Files in This Session
- **SESSION_SUMMARY.md** - Complete session data and status
- **README.md** - This file
- **cluster_status.txt** - Cluster resources snapshot
- **jobs_status.txt** - Current SLURM jobs
- **disk_usage.txt** - Disk usage summary

## How to Review
1. Open \SESSION_SUMMARY.md\ for complete overview
2. Check \jobs_status.txt\ for SLURM queue
3. Review \cluster_status.txt\ for resource availability

## Session Management
- **Session Rule:** SESSION_NAMING_RULE.md in parent directory
- **All sessions:** F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\ClaudeSshSession\
- **Next session:** Run \.\generate-session-summary.ps1\ with different JobId

## Monitoring Commands

### Check specific job
\\\ash
ssh eden-cluster "scontrol show job 1190904"
\\\

### View full training log
\\\ash
ssh eden-cluster "tail -f '********************************************************* SSH server at Warsaw University of Technology Faculty of Mathematics and Information Science System.Management.Automation.RemoteException Individuals using this computer system without authority, or in  excess of  their  authority, are subject to having all  of  their activities  on this  system monitored  and recorded   by   system  personnel.  In   the   course  of monitoring  individuals improperly  using this system, or in the  course of system  maintenance, the  activities of authorized  users  may  also be  monitored.  Anyone using this system  expressly consents to such monitoring and is advised that if such monitoring reveals possible criminal activity , system personnel may  provide the  evidence of such monitoring to law enforcement officials. ********************************************************* /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_ddp_6gpu_dgx2_500ep_1190904.log'"
\\\

### Get cluster status
\\\ash
ssh eden-cluster "sfree"
\\\

### Find all logs for JobId
\\\ash
ssh eden-cluster "find /mnt/evafs/faculty/home/bpiotrowski/DETR/logs -name '*1190904*'"
\\\

---
**Next Step:** Review SESSION_SUMMARY.md for details
