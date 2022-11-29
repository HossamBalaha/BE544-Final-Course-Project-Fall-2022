$(document).ready(function(){let y=0,H=10,w=0,p=$("#clsHist-body"),d=$("#loading-blocker");function C(){$('button[id^="row-delete-clsHist-"]').off().click(function(g){Swal.fire({title:"Are you sure?",text:"You won't be able to revert this!",icon:"warning",showCancelButton:!0,confirmButtonColor:"#3085d6",cancelButtonColor:"#d33",confirmButtonText:"Yes, delete it!"}).then(u=>{if(u.isConfirmed){d.show();let t=$(this).attr("id").split("-").slice(3).join("-");$.ajax({url:"/api/classification/delete",type:"POST",data:{target:t,csrf_token:CSRF_TOKEN},success:function(e){e.is_success?Swal.fire({icon:"success",title:"Deleted",text:e.message}):Swal.fire({icon:"error",title:"Oops...",text:e.message})},error:function(e){Swal.fire({icon:"error",title:"Oops...",text:e.message})},complete:function(){d.hide(),$("#clsHist-refresh").click()}})}})}),$('button[id^="row-view-clsHist-"][id$="-init-configurations"]').off().click(function(g){let o=$(this).attr("id").split("-").slice(3,-2).join("-");d.show(),$.ajax({url:"/api/classification/configs/init",type:"POST",data:{target:o,csrf_token:CSRF_TOKEN},success:function(t){if(t.is_success){let e=`modal-clsHist-${o}-common`,s=$(document.getElementById(e));s.modal("show");let r='<ul class="list-group">';for(let c=0;c<t.result.length;c++){let i=t.result[c];console.log(i),r+=`<li class="list-group-item my-0 py-1 d-flex justify-content-between align-items-center">
    ${i[0]}
    <span class="badge bg-primary rounded-pill">${i[1]}</span>
  </li>`}r+="</ul>",s.find(`#${e}-body`).empty().html(r),s.find(`#${e}-title`).text("Initial Configurations"),s.find(".modal-dialog").removeClass("modal-xl").removeClass("modal-full-screen").addClass("modal-lg")}else Swal.fire({icon:"error",title:"Oops...",text:t.message})},error:function(t){Swal.fire({icon:"error",title:"Oops...",text:t.message})},complete:function(){d.hide()}})}),$('button[id^="row-view-clsHist-"][id$="-post-configurations"]').off().click(function(g){let o=$(this).attr("id").split("-").slice(3,-2).join("-");d.show(),$.ajax({url:"/api/classification/configs/post",type:"POST",data:{target:o,csrf_token:CSRF_TOKEN},success:function(t){if(t.is_success){let e=`modal-clsHist-${o}-common`,s=$(document.getElementById(e));s.modal("show");let r='<ul class="list-group">';for(let c=0;c<t.result.length;c++){let i=t.result[c];console.log(i),r+=`<li class="list-group-item my-0 py-1 d-flex justify-content-between align-items-center">
    ${i[0]}
    <span class="badge bg-primary rounded-pill">${i[1]}</span>
  </li>`}r+="</ul>",s.find(`#${e}-body`).empty().html(r),s.find(`#${e}-title`).text("Post Configurations"),s.find(".modal-dialog").removeClass("modal-xl").removeClass("modal-full-screen").addClass("modal-lg")}else Swal.fire({icon:"error",title:"Oops...",text:t.message})},error:function(t){Swal.fire({icon:"error",title:"Oops...",text:t.message})},complete:function(){d.hide()}})}),$('button[id^="row-view-clsHist-"][id$="-history"]').off().click(function(g){let o=$(this).attr("id").split("-").slice(3,-1).join("-");d.show(),$.ajax({url:"/api/classification/history",type:"POST",data:{target:o,csrf_token:CSRF_TOKEN},success:function(t){if(t.is_success){let e=`modal-clsHist-${o}-common`,s=$(document.getElementById(e));s.modal("show");let r='<th scope="col">No.</th>',c=t.result.columns;for(let n=0;n<c.length;n++)r+=`<th scope="col">${c[n].toUpperCase()}</th>`;let i=`<table class="table table-striped table-hover table-sm table-bordered">
              <thead><tr>${r}</tr></thead><tbody>`;for(let n=0;n<t.result.history.length;n++){let b=t.result.history[n];i+=`<tr><th scope="row">${n+1}</th>`;for(let f=0;f<b.length;f++)i+=`<td>${b[f]}</td>`;i+="</tr>"}i+="</tbody></table>",s.find(`#${e}-body`).empty().html(i),s.find(`#${e}-title`).text("History"),s.find(".modal-dialog").removeClass("modal-lg").removeClass("modal-xl").addClass("modal-full-screen")}else Swal.fire({icon:"error",title:"Oops...",text:t.message})},error:function(t){Swal.fire({icon:"error",title:"Oops...",text:t.message})},complete:function(){d.hide()}})}),$('button[id^="row-view-clsHist-"][id$="-evaluations"]').off().click(function(g){let o=$(this).attr("id").split("-").slice(3,-1).join("-");d.show(),$.ajax({url:"/api/classification/evaluations",type:"POST",data:{target:o,csrf_token:CSRF_TOKEN},success:function(t){if(t.is_success){let e=`modal-clsHist-${o}-common`,s=$(document.getElementById(e));s.modal("show");let r='<th scope="col">No.</th>',c=t.result.columns;for(let n=0;n<c.length;n++)r+=`<th scope="col">${c[n].toUpperCase()}</th>`;let i=`<table class="table table-striped table-hover table-sm table-bordered">
              <thead><tr>${r}</tr></thead><tbody>`;for(let n=0;n<t.result.evaluations.length;n++){let b=t.result.evaluations[n];i+=`<tr><th scope="row">${n+1}</th>`;for(let f=0;f<b.length;f++)i+=`<td>${b[f]}</td>`;i+="</tr>"}i+="</tbody></table>",s.find(`#${e}-body`).empty().html(i),s.find(`#${e}-title`).text("Evaluations"),s.find(".modal-dialog").removeClass("modal-full-screen").removeClass("modal-lg").addClass("modal-xl")}else Swal.fire({icon:"error",title:"Oops...",text:t.message})},error:function(t){Swal.fire({icon:"error",title:"Oops...",text:t.message})},complete:function(){d.hide()}})}),$('button[id^="row-view-clsHist-"][id$="-history-plot"]').off().click(function(g){let o=$(this).attr("id").split("-").slice(3,-2).join("-");d.show(),$.ajax({url:"/api/classification/history/plot",type:"POST",data:{target:o,csrf_token:CSRF_TOKEN},success:function(t){if(t.is_success){let e=`modal-clsHist-${o}-common`,s=$(document.getElementById(e));s.modal("show");let r=`<img src="data:image/jpeg;base64,${t.result}" class="img-fluid" alt="History Plot">`;s.find(`#${e}-body`).empty().html(r),s.find(`#${e}-title`).text("History"),s.find(".modal-dialog").removeClass("modal-lg").removeClass("modal-xl").addClass("modal-full-screen")}else Swal.fire({icon:"error",title:"Oops...",text:t.message})},error:function(t){Swal.fire({icon:"error",title:"Oops...",text:t.message})},complete:function(){d.hide()}})}),$('button[id^="message-clsHist-"]').off().click(function(g){let o=$(this).attr("data-message");Swal.fire({icon:"error",title:"Failure Message",text:o})})}let v=()=>{$("table").DataTable().destroy(),d.show(),p.append(`<tr id="tr-loader"><td colspan="8" class="text-center align-middle">
          <div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></td></tr>`),$.ajax({url:"/api/classification/browse",type:"GET",data:{from:y,step:H},success:function(m){if(y===0&&p.empty(),$("#tr-loader").remove(),m.is_success){let h="";m.result.elements.forEach((l,S)=>{let x=`<button class="btn btn-sm btn-danger" type="button" 
              title="View Failure Message" 
              data-message="${l.message}"
              id="message-clsHist-${l.storeName}">
                <i class="fa fa-triangle-exclamation"></i>
              </button>`,a=$('<tr id="row"></tr>');a.append(`<td class="text-center align-middle">${++w}</td>`),a.append(`<td class="text-center align-middle">${l.name}</td>`),a.append(`<td class="text-center align-middle">
            ${l.isSuccess?"Completed":"Failed"}
            ${l.isSuccess?"":x}
            </td>`),a.append(`<td class="text-center align-middle">${l.createdAt}</td>`),a.append(`<td class="text-center align-middle">${l.updatedAt}</td>`),a.append(`<td class="text-center align-middle">
              <button class="btn btn-sm btn-primary" type="button" 
              title="View Classification Initial Configurations"
                id="row-view-clsHist-${l.storeName}-init-configurations">
                <i class="fa fa-eye"></i>
              </button></td>
            `),l.isSuccess?(a.append(`<td class="text-center align-middle">
              <button class="btn btn-sm btn-primary" type="button" 
              title="View Classification Post Configurations"
                id="row-view-clsHist-${l.storeName}-post-configurations">
                <i class="fa fa-eye"></i>
              </button></td>
            `),a.append(`<td class="text-center align-middle">
              <button class="btn btn-sm btn-primary" type="button" 
              title="View Classification History"
                id="row-view-clsHist-${l.storeName}-history">
                <i class="fa fa-eye"></i>
              </button></td>
            `),a.append(`<td class="text-center align-middle">
              <button class="btn btn-sm btn-primary" type="button" 
              title="View Classification History Plot"
                id="row-view-clsHist-${l.storeName}-history-plot">
                <i class="fa fa-eye"></i>
              </button></td>
            `),a.append(`<td class="text-center align-middle">
              <button class="btn btn-sm btn-primary" type="button" 
              title="View Classification Evaluations"
                id="row-view-clsHist-${l.storeName}-evaluations">
                <i class="fa fa-eye"></i>
              </button></td>
            `)):(a.append('<td class="text-center align-middle"></td>'),a.append('<td class="text-center align-middle"></td>'),a.append('<td class="text-center align-middle"></td>'),a.append('<td class="text-center align-middle"></td>')),a.append(`<td class="text-center align-middle">
              <button class="btn btn-sm btn-danger" type="button" 
              title="Remove Classification Record"
                id="row-delete-clsHist-${l.storeName}">
                <i class="fa fa-trash"></i>
              </button></td>
            `),p.append(a),h+=`<div class="modal fade" id="modal-clsHist-${l.storeName}-common" 
                tabindex="-1" aria-labelledby="modal-clsHist-${l.storeName}-common" aria-hidden="true">
                <div class="modal-dialog modal-dialog-centered modal-dialog-scrollable modal-lg">
                  <div class="modal-content">
                    <div class="modal-header">
                      <h5 class="modal-title" id="modal-clsHist-${l.storeName}-common-title"></h5>
                      <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body text-center" id="modal-clsHist-${l.storeName}-common-body">
                      <div class="row">
                        <div class="col-12">
                          <div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div>
                        </div>
                      </div>
                    </div>
                    <div class="modal-footer">
                      <button type="button" class="btn btn-secondary btn-sm" data-bs-dismiss="modal">Close</button>
                    </div>
                  </div>
                </div>
              </div>`}),$("body").append(h),$("#clsHistCountFiles").html(`<span class="text-white w-auto bg-secondary badge">
            Showing ${w} from the ${m.result.count} records.</span>`)}else p.append(`<tr><td colspan="8" class="text-center align-middle">
                <h4 class="text-danger">${m.message}</h4></td></tr>`)},error:function(m){Swal.fire({icon:"error",title:"Oops...",text:m.message})},complete:function(){d.hide(),$("table").DataTable({bDestroy:!0,initComplete:function(m,h){$("table").wrap("<div style='overflow:auto; width:100%;position:relative;'></div>")}}).draw(),C()}})};v(),$("#clsHist-refresh").click(function(){p.empty(),w=0,y=0,v()}),$("#loadMoreClsHistItems").off().click(function(){y+=H,v()})});