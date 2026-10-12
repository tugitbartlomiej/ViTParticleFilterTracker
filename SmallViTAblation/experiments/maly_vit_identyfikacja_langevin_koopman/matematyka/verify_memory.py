"""Exact checks of finite-memory elimination, AR covariance and rank examples.
Run with Python + SymPy; writes only beside this script.
"""
import json
import platform
from pathlib import Path
import sympy as sp
from sympy.stats import Normal, covariance, variance


def check_system(A, B, C, E, D, F, x0, h0, inputs, label):
    nx, nh = A.rows, E.rows
    xs, hs = [x0], [h0]
    records = []
    for k, uk in enumerate(inputs):
        memory_x = sp.zeros(nx, 1)
        memory_u = sp.zeros(nx, 1)
        hidden_sum = E**k * h0
        for j in range(k):
            hidden_sum += E**(k - 1 - j) * (C * xs[j] + F * inputs[j])
            memory_x += B * E**(k - 1 - j) * C * xs[j]
            memory_u += B * E**(k - 1 - j) * F * inputs[j]
        predicted = A * xs[k] + memory_x + D * uk + memory_u + B * E**k * h0
        actual = A * xs[k] + B * hs[k] + D * uk
        dx = (predicted - actual).applyfunc(sp.expand)
        dh = (hidden_sum - hs[k]).applyfunc(sp.expand)
        assert dx == sp.zeros(nx, 1), (label, k, dx)
        assert dh == sp.zeros(nh, 1), (label, k, dh)
        records.append({'k': k, 'hidden_residual_zero': True, 'observed_residual_zero': True})
        xs.append(actual)
        hs.append(C * xs[k] + E * hs[k] + F * uk)
    return {'name': label, 'shape': {'x': nx, 'h': nh, 'u': D.cols}, 'checks': records}


def main():
    checks = []
    a,b,c,e,d,f,x0,h0 = sp.symbols('a b c e d f x0 h0')
    uu = sp.symbols('u0:7')
    checks.append(check_system(*[sp.Matrix([[v]]) for v in [a,b,c,e,d,f]],
        sp.Matrix([x0]), sp.Matrix([h0]), [sp.Matrix([u]) for u in uu], 'fully_symbolic_scalar'))
    # Noncommuting symbolic 2x2 matrices, independent symbolic inputs and initial state.
    mats = [sp.Matrix(2,2,sp.symbols(f'{name}0:4')) for name in ['A','B','C','E','D','F']]
    checks.append(check_system(*mats, sp.Matrix(sp.symbols('xs0:2')),
        sp.Matrix(sp.symbols('hs0:2')), [sp.Matrix(sp.symbols(f'us{k}_0:2')) for k in range(3)],
        'fully_symbolic_noncommuting_2x2'))
    R = sp.Rational
    A=sp.Matrix([[R(1,2),R(1,3)],[-R(1,4),R(2,3)]])
    B=sp.Matrix([[1,2],[-1,1]])
    C=sp.Matrix([[2,-1],[1,3]])
    D=sp.Matrix([1,-2]); F=sp.Matrix([2,1])
    for tag,E in [('stable',sp.Matrix([[R(1,3),R(1,4)],[-R(1,5),R(1,2)]])),
                  ('nilpotent',sp.Matrix([[0,1],[0,0]]))]:
        checks.append(check_system(A,B,C,E,D,F,sp.Matrix([2,-1]),sp.Matrix([1,3]),
            [sp.Matrix([R(k-2,k+1)]) for k in range(9)], 'exact_rational_2x2_'+tag))
    # Stationary Gaussian AR(1), with 0<rho<1, q>0 and mutually independent normals.
    rho=sp.symbols('rho', positive=True)
    q=sp.symbols('q', positive=True)
    z0=Normal('z0',0,1); z1=Normal('z1',0,1)
    eta0=sp.sqrt(q/(1-rho**2))*z0
    eta1=rho*eta0+sp.sqrt(q)*z1
    lag1=sp.simplify(covariance(eta1,eta0))
    stationary_var=sp.simplify(variance(eta1))
    assert sp.simplify(lag1-rho*q/(1-rho**2))==0
    assert sp.simplify(stationary_var-q/(1-rho**2))==0
    example=sp.simplify(lag1.subs({rho:R(1,2),q:R(3,4)}))
    assert example==R(1,2)
    rank_examples=[]
    for tag,T,G in [('minimal',sp.Matrix([[1,1],[1,0]]),sp.Matrix([0,1])),
                    ('hidden_decoupled',sp.diag(1,R(1,2)),sp.Matrix([1,0]))]:
        H=sp.Matrix([[1,0]])
        O=H.col_join(H*T); K=G.row_join(T*G)
        rank_examples.append({'name':tag,'state_dimension':2,'observability_rank':O.rank(),
            'controllability_rank':K.rank(),'T':str(T),'G':str(G),'H':str(H)})
    result={'sympy_version':sp.__version__,'python_version':platform.python_version(),
        'arithmetic':'exact symbolic / rational; no floats','systems':checks,
        'gaussian_ar1':{'assumptions':'0 < rho < 1, q > 0; z0,z1 independent standard normal',
            'lag1_covariance':str(lag1),'stationary_variance':str(stationary_var),
            'rho_half_q_three_quarters_covariance':str(example),
            'conclusion':'nonzero lag covariance excludes white noise; Gaussian marginal alone does not imply white'},
        'rank_examples':rank_examples,'all_checks_passed':True}
    out=Path(__file__).with_name('sympy_wynik.json')
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'all_checks_passed':True,'systems':len(checks),
        'observed_hidden_pairs':sum(len(v['checks']) for v in checks),'output':str(out)}))

if __name__=='__main__': main()
