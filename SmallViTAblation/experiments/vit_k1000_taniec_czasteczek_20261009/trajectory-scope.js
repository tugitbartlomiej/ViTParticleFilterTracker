'use strict';
function parseTrajectoryImages(value,maximum=24,total=1500){
 const result=[],seen=new Set();
 if(typeof value!=='string'||!value.trim())throw Error('Podaj numery zdjęć, np. 12, 40-45');
 for(const raw of value.split(',')){
  const token=raw.trim();let numbers;
  if(/^\d+$/.test(token))numbers=[Number(token)];
  else if(/^\d+-\d+$/.test(token)){const [a,b]=token.split('-').map(Number);if(a>b)throw Error('Zakres zdjęć musi rosnąć');numbers=Array.from({length:b-a+1},(_,i)=>a+i);}
  else throw Error('Niepoprawna lista zdjęć');
  for(const image of numbers){if(image<0||image>=total)throw Error('Numer zdjęcia musi należeć do 0..1499');if(!seen.has(image)){seen.add(image);result.push(image);}if(result.length>maximum)throw Error(`Pełne ślady można wczytać dla najwyżej ${maximum} zdjęć`);}
 }
 return result;
}
function trajectoryInterval(steps,step){let a=Math.max(0,steps.findLastIndex(s=>s<=step)),b=Math.min(a+1,steps.length-1);return {a,b,u:b===a?0:(step-steps[a])/(steps[b]-steps[a])};}
function trajectoryPlan(scope,selectedCount,tail){
 if(scope==='selected'){if(!Number.isInteger(selectedCount)||selectedCount<1||selectedCount>24)throw Error('Wybierz od 1 do 24 zdjęć');return {scope,particles:selectedCount*600,segments:selectedCount*600*100,tail:100,fullHistory:true,sampling:false};}
 if(scope==='global'){if(!Number.isInteger(tail)||tail<1||tail>10)throw Error('Ogon musi mieć od 1 do 10 przedziałów');return {scope,particles:900000,segments:900000*tail,tail,fullHistory:false,sampling:false};}
 throw Error('Nieznany zakres trajektorii');
}
