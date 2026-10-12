"""Exact controlled-memory and joint block-ridge checks, plus local Lean compilation.
Only local synthetic arithmetic. No measurements, KF, FDT or cloud calls.
"""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sympy as s

ROOT=Path(__file__).resolve().parent
LEAN=Path('C:/Users/bartl/.elan/toolchains/leanprover--lean4---v4.28.0/bin/lean.exe')

def zero(M): return M.applyfunc(s.expand)==s.zeros(*M.shape)

def matrix_elimination():
    # All entries independent symbolic scalars: products are not commuted.
    A,B,C,E,D,F=[s.Matrix(2,2,s.symbols(f'{name}0:4')) for name in ['A','B','C','E','D','F']]
    x0=s.Matrix(s.symbols('x0:2')); h0=s.Matrix(s.symbols('h0:2'))
    U=[s.Matrix(s.symbols(f'u{k}_0:2')) for k in range(4)]
    X=[x0]; H=[h0]; checks=[]
    for k in range(4):
        sx=s.zeros(2,1); su=s.zeros(2,1)
        for lag in range(1,k+1):
            sx += B*E**(lag-1)*C*X[k-lag]
            su += B*E**(lag-1)*F*U[k-lag]
        direct=A*X[k]+B*H[k]+D*U[k]
        split=A*X[k]+D*U[k]+sx+su+B*E**k*h0
        assert zero(direct-split)
        H.append(E*H[k]+C*X[k]+F*U[k]); X.append(direct)
        checks.append({'k':k,'split_elimination_exact':True,'max_input_index_in_memory':k-1,
            'current_input_index_in_direct_term':k})
    # At t=3, perturb u_3 and leave all strictly historical inputs untouched.
    delta=s.Matrix(s.symbols('delta0:2'))
    prefix_memory=sum((B*E**(2-j)*F*U[j] for j in range(3)),s.zeros(2,1))
    altered=list(U); altered[3]=U[3]+delta
    altered_memory=sum((B*E**(2-j)*F*altered[j] for j in range(3)),s.zeros(2,1))
    assert zero(prefix_memory-altered_memory)
    assert zero((A*X[3]+B*H[3]+D*altered[3])-(A*X[3]+B*H[3]+D*U[3])-D*delta)
    return {'dimensions':{'x':2,'h':2,'u':2},'all_entries_independent_symbolic':True,
        'checks':checks,'changing_u_t_preserves_memory_t':True,
        'changing_u_t_changes_x_next_by':'D*delta_u_t; generally nonzero'}

def block_ridge():
    # Row sample convention. At origin t only x_t and u_{<t} are available.
    # Four feature blocks: drift x_t, recent u_{t-1}, state memory x_{t-1}, older input u_{t-2}.
    X=[s.Matrix([k+1,(k*k+3)%7-2]) for k in range(13)]
    U=[s.Integer(k**3%11-5) for k in range(13)]
    rows=[]; targets=[]
    for t in range(2,12):
        rows.append([*X[t],U[t-1],*X[t-1],U[t-2]])
        targets.append(list(X[t+1]))
    Z=s.Matrix(rows); Y=s.Matrix(targets)
    groups=[('drift',0,2,1),('recent_input',2,3,4),('state_memory',3,5,9),('older_input_memory',5,6,16)]
    L=s.diag(1,1,4,9,9,16)
    S=s.diag(1,1,2,3,3,4)
    W=(Z.T*Z+L).inv()*Z.T*Y
    assert zero((Z.T*Z+L)*W-Z.T*Y)
    assert zero(Z.T*(Z*W-Y)+L*W)
    Q=Z*S.inv()
    W_scaled=S.inv()*(Q.T*Q+s.eye(Z.cols)).inv()*Q.T*Y
    assert zero(W-W_scaled)
    # Gram cross-block terms remain in the joint normal equations.
    separately=[]
    for _,lo,hi,penalty in groups:
        zg=Z[:,lo:hi]
        separately.append((zg.T*zg+penalty*s.eye(hi-lo)).inv()*zg.T*Y)
    W_separate=s.Matrix.vstack(*separately)
    differs=not zero(W-W_separate)
    assert differs
    return {'Z_shape':list(Z.shape),'Y_shape':list(Y.shape),'feature_rank':Z.rank(),
        'blocks':[{'name':name,'columns_halfopen':[lo,hi],'penalty':penalty} for name,lo,hi,penalty in groups],
        'normal_equation_residual_zero':True,'objective_gradient_zero':True,
        'column_scaling_identity_exact':True,'independent_block_fits_differ':differs,
        'largest_used_input_index':'t-1','target_index':'t+1',
        'claim':'algebraic identity for exact synthetic design; no forecast validity or Koopman closure claim'}

def symbolic_block_normal_equation():
    # Two scalar blocks: generic symmetric Gram and distinct penalties.
    g11,g12,g22,l1,l2,r1,r2=s.symbols('g11 g12 g22 l1 l2 r1 r2')
    H=s.Matrix([[g11+l1,g12],[g12,g22+l2]])
    rhs=s.Matrix([r1,r2]); W=H.inv()*rhs
    residual=(H*W-rhs).applyfunc(s.cancel)
    assert residual==s.zeros(2,1)
    return {'symbolic_generic_two_blocks':True,'normal_equation_residual_zero':True,
        'condition':'det(Gram+Lambda) != 0; positive penalties suffice for Gram=Z.T*Z',
        'determinant':str(s.factor(H.det()))}

def main():
    result={'sympy_version':s.__version__,'python_version':platform.python_version(),
        'matrix_elimination':matrix_elimination(),'joint_block_ridge':block_ridge(),
        'symbolic_block_ridge':symbolic_block_normal_equation()}
    commands={'lean_version':[str(LEAN),'--version'],'lean':[str(LEAN),str(ROOT/'ControlledMemory.lean')]}
    for name,command in commands.items():
        p=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        result[name]={'command':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
        assert p.returncode==0,(name,p.stdout,p.stderr)
    result['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['ControlledMemory.lean','controlled_verify.py']}
    result['all_checks_passed']=True
    path=ROOT/'controlled_verification_result.json'
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'all_checks_passed':True,'output':str(path)}))

if __name__=='__main__': main()
