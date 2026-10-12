const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const context=vm.createContext({console,Number,Array,Set,Error});
vm.runInContext(fs.readFileSync(__dirname+'/trajectory-scope.js','utf8'),context);

assert.deepEqual(Array.from(context.parseTrajectoryImages('12, 40-42, 12, 3')),[12,40,41,42,3]);
for(const bad of ['', '3-1', '-2', '1500', '1,a'])assert.throws(()=>context.parseTrajectoryImages(bad));
assert.throws(()=>context.parseTrajectoryImages('0-24',24));
assert.deepEqual(JSON.parse(JSON.stringify(context.trajectoryInterval([0,5,10],7.5))),{a:1,b:2,u:.5});
assert.deepEqual(JSON.parse(JSON.stringify(context.trajectoryInterval([0,5,10],10))),{a:2,b:2,u:0});
assert.deepEqual(JSON.parse(JSON.stringify(context.trajectoryPlan('selected',3,10))),{scope:'selected',particles:1800,segments:180000,tail:100,fullHistory:true,sampling:false});
assert.deepEqual(JSON.parse(JSON.stringify(context.trajectoryPlan('global',0,3))),{scope:'global',particles:900000,segments:2700000,tail:3,fullHistory:false,sampling:false});
assert.throws(()=>context.trajectoryPlan('global',0,11));
console.log('PASS: image ranges, selection cap and trajectory time interval.');
