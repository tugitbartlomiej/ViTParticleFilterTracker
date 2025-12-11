# Session Summary: {SESSION_NAME}

## Metadata
- **Type:** ssh
- **Date:** {DATE}
- **Time:** {TIME} UTC
- **Duration:** {DURATION}
- **Status:** {STATUS}

## Objective
{OBJECTIVE}

## Context
{CONTEXT}

## SSH Connection Details
- **Cluster:** {CLUSTER_NAME}
- **User:** {USERNAME}
- **Node:** {NODE_NAME}
- **Job ID:** {JOB_ID}

## Actions Taken

### 1. Cluster Status Check
```bash
{CLUSTER_STATUS_COMMAND}
```

**Output:**
```
{CLUSTER_STATUS_OUTPUT}
```

### 2. Job Submission/Monitoring
```bash
{JOB_COMMAND}
```

**Job Details:**
- Job ID: {JOB_ID}
- Partition: {PARTITION}
- Nodes: {NUM_NODES}
- GPUs: {NUM_GPUS} x {GPU_TYPE}
- Status: {JOB_STATUS}

### 3. Log Monitoring
```bash
{LOG_COMMAND}
```

**Key Log Entries:**
```
{LOG_EXCERPT}
```

### 4. File Operations
- Uploaded: {UPLOADED_FILES}
- Downloaded: {DOWNLOADED_FILES}
- Modified: {MODIFIED_FILES}

## Results

### Training Progress
- **Current Epoch:** {CURRENT_EPOCH}
- **Total Epochs:** {TOTAL_EPOCHS}
- **Training Loss:** {TRAIN_LOSS}
- **Validation Loss:** {VAL_LOSS}
- **Time Remaining:** {ETA}

### Resource Usage
- **CPU Usage:** {CPU_USAGE}
- **GPU Usage:** {GPU_USAGE}
- **Memory Usage:** {MEMORY_USAGE}
- **Disk Usage:** {DISK_USAGE}

### Checkpoints
- **Latest Checkpoint:** {CHECKPOINT_PATH}
- **Checkpoint Epoch:** {CHECKPOINT_EPOCH}
- **Checkpoint Size:** {CHECKPOINT_SIZE}
- **Archived:** {ARCHIVED_STATUS}

## Issues Encountered

### Issue 1: {ISSUE_TITLE}
**Description:** {ISSUE_DESCRIPTION}
**Resolution:** {ISSUE_RESOLUTION}
**Status:** {ISSUE_STATUS}

## Conclusions
{CONCLUSIONS}

## Next Steps
- [ ] {NEXT_STEP_1}
- [ ] {NEXT_STEP_2}
- [ ] {NEXT_STEP_3}

## Files Generated
- `cluster_status.txt` - Cluster resource snapshot
- `job_details.txt` - SLURM job information
- `logs/training_{JOB_ID}.log` - Training log excerpt

## Commands Used

### Connect to cluster
```bash
ssh {USERNAME}@{CLUSTER_HOST}
```

### Check job status
```bash
squeue -u {USERNAME}
scontrol show job {JOB_ID}
```

### Monitor training log
```bash
tail -f {LOG_PATH}
```

### Download checkpoint
```bash
scp {USERNAME}@{CLUSTER_HOST}:{CHECKPOINT_PATH} ./local_path/
```

## Related Work
- **Previous session:** {PREVIOUS_SESSION}
- **Next session:** {NEXT_SESSION}
- **Related analysis:** {RELATED_ANALYSIS}
- **Checkpoint archive:** {CHECKPOINT_ARCHIVE}

---

**Session Created:** {CREATED_TIMESTAMP}
**Last Updated:** {UPDATED_TIMESTAMP}
