"""Stan przed następną epoką: zachowuje historię aktualizacji, zeruje kursor paczek.

Lightning 2.6.6 może pozostawić kursor zakończonego dataloadera po fit(max_steps).
Pełny checkpoint zapisany wtedy z is_last_batch=True powtarza zamknięcie epoki
przy resume. Kanonizacja dotyczy wyłącznie potwierdzonej pełnej granicy epoki.
Nie resetuje wag, momentów optymalizatora, harmonogramu ani RNG.
"""
from copy import deepcopy

def canonical_boundary(checkpoint,steps_epoch):
    step=int(checkpoint['global_step'])
    if step<=0 or step%steps_epoch:raise ValueError('Not a complete epoch boundary')
    fit=checkpoint['loops']['fit_loop']
    n=step//steps_epoch
    if any(value!=n for counters in fit['epoch_progress'].values() for value in counters.values()):
        raise ValueError('Epoch and optimizer step counters disagree')
    checkpoint['experiment_raw_loops']=deepcopy(checkpoint['loops'])
    # All total optimizer counters are retained. Only the next-epoch cursor is cleared.
    def reset_current(value):
        if not isinstance(value,dict):return
        if 'current' in value and isinstance(value['current'],dict):
            value['current']={k:0 for k in value['current']}
        if 'is_last_batch' in value:value['is_last_batch']=False
        for child in value.values():reset_current(child)
    for key,value in fit.items():
        if key.startswith('epoch_loop.') and 'progress' in key:reset_current(value)
    # Validation counters are per epoch, not lifetime training counters.
    v=fit['epoch_loop.val_loop.batch_progress']
    v['total']={k:0 for k in v['total']}
    checkpoint['experiment_boundary']='before_next_epoch_v1'
    return checkpoint
