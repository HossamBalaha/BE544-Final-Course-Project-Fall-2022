$(document).ready(function(){let c=0,b=10,n=0,h=$("#wsi-upload"),d=$("#wsi-body"),s=$("#loading-blocker");h.click(function(){let i=$('<input type="file">');i.change(function(){s.show();let e=i[0].files,a=new FormData;a.append("file",e[0]),a.append("csrf_token",CSRF_TOKEN),$.ajax({url:"/api/wsi/upload",type:"POST",data:a,processData:!1,contentType:!1,success:function(t){t.is_success===!1?Swal.fire({icon:"error",title:"Oops...",text:t.message}):(Swal.fire({icon:"success",title:"Success",text:t.message}),$("#wsi-refresh").click())},error:function(t){Swal.fire({icon:"error",title:"Oops...",text:"Error uploading the file"}),console.log(t)},complete:function(){s.hide(),$("#wsi-refresh").click()}})}),i.click()});function g(){$('button[id^="row-delete-wsi-"]').off().click(function(a){Swal.fire({title:"Are you sure?",text:"You won't be able to revert this!",icon:"warning",showCancelButton:!0,confirmButtonColor:"#3085d6",cancelButtonColor:"#d33",confirmButtonText:"Yes, delete it!"}).then(t=>{if(t.isConfirmed){s.show();let l=$(this);$.ajax({url:"/api/wsi/delete",type:"POST",data:{item:l.attr("id"),csrf_token:CSRF_TOKEN},success:function(o){o.is_success?Swal.fire({icon:"success",title:"Deleted",text:o.message}):Swal.fire({icon:"error",title:"Oops...",text:o.message})},error:function(o){Swal.fire({icon:"error",title:"Oops...",text:o.message})},complete:function(){s.hide(),$("#wsi-refresh").click()}})}})}),$('button[id^="row-dzi-"]').off().click(function(a){a.preventDefault();let l=$(this).attr("id"),o=$(document.getElementById(l.replace("row-dzi-","row-conv-wsi-")));o.modal("show");let u=$(document.getElementById(l.replace("row-dzi-","row-conv-wsi-")+"-submit"));u.off().click(function(x){x.preventDefault(),Swal.fire({title:"Are you sure?",text:"Once the Generate DZI process is submitted, it will take a while to complete. The process may take long time to complete. You can check the progress of the process in the DZI Handler page.",icon:"warning",showCancelButton:!0,confirmButtonColor:"#3085d6",cancelButtonColor:"#d33",confirmButtonText:"Yes, convert it!"}).then(y=>{if(y.isConfirmed){s.show();let m=$(document.getElementById(l.replace("row-dzi-","row-conv-wsi-")+"-size")),f=$(document.getElementById(l.replace("row-dzi-","row-conv-wsi-")+"-overlap")),w=$(document.getElementById(l.replace("row-dzi-","row-conv-wsi-")+"-title")),v=$(document.getElementById(l.replace("row-dzi-","row-conv-wsi-")+"-close"));m.prop("disabled",!0),f.prop("disabled",!0),w.prop("disabled",!0),u.prop("disabled",!0),v.prop("disabled",!0),$.ajax({url:"/api/wsi/dzi",type:"POST",data:{item:l,size:m.val(),overlap:f.val(),title:w.val(),csrf_token:CSRF_TOKEN},success:function(r){r.is_success?Swal.fire({icon:"success",title:"Success",text:r.message}):Swal.fire({icon:"error",title:"Oops...",text:r.message})},error:function(r){Swal.fire({icon:"error",title:"Oops...",text:r.message})},complete:function(){m.prop("disabled",!1),f.prop("disabled",!1),w.prop("disabled",!1),u.prop("disabled",!1),v.prop("disabled",!1),o.modal("hide"),s.hide()}})}})})})}let p=()=>{$("table").DataTable().destroy(),s.show(),d.append(`<tr id="tr-loader"><td colspan="8" class="text-center align-middle">
                <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
              </div></td></tr>`),$.ajax({url:"/api/wsi/browse",type:"GET",data:{from:c,step:b},success:function(i){c===0&&d.empty(),$("#tr-loader").remove(),i.is_success?(i.result.elements.forEach((e,a)=>{let t=$('<tr id="row"></tr>');t.append(`<td class="text-center align-middle">${++n}</td>`),t.append(`<td class="text-center align-middle">${e.name}</td>`),e.size>1024*1024*1024?t.append(`<td class="text-center align-middle">${(e.size/(1024*1024*1024)).toFixed(2)} GB</td>`):e.size>1024*1024?t.append(`<td class="text-center align-middle">${(e.size/(1024*1024)).toFixed(2)} MB</td>`):e.size>1024?t.append(`<td class="text-center align-middle">${(e.size/1024).toFixed(2)} KB</td>`):t.append(`<td class="text-center align-middle">${e.size} B</td>`),t.append(`<td class="text-center align-middle">${e.modifiedAt}</td>`),t.append(`<td class="text-center align-middle">${e.createdAt}</td>`),t.append(`<td class="text-center align-middle">${e.accessedAt}</td>`);let l="";l=`<button class="btn btn-sm btn-primary" title="Generate DZI" id="row-dzi-${e.path}">
                <i class="fa fa-person-walking-arrow-loop-left"></i>
              </button>
              <button class="btn btn-sm btn-danger" title="Delete WSI" id="row-delete-wsi-${e.path}">
                <i class="fa fa-trash"></i>
              </button>`,t.append(`<td class="text-center align-middle"><div>${l}</div></td>`),d.append(t);let o=`<div class="modal fade" id="row-conv-wsi-${e.path}" tabindex="-1"
                    aria-labelledby="row-conv-wsi-${e.path}-label" aria-hidden="true">
                    <div class="modal-dialog mx-auto modal-lg modal-scrollable modal-dialog-centered" role="document">
                      <div class="modal-content">
                        <div class="modal-header text-center">
                          <h1 class="modal-title fs-5 text-center" id="row-conv-wsi-${e.path}-label">
                            Convert WSI to DZI Configurations
                          </h1>
                          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                        <div class="alert alert-dismissible alert-info">
                          <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                              Once the Generate DZI process is submitted, it will take a while to complete.
                              The process may take long time to complete. You can check the progress of the process
                              in the DZI Handler page.
                          </div>
                
                          <div class="row mb-2">
                            <div class="col-12 col-md-6 col-lg-4">
                              <label for="row-conv-wsi-${e.path}-title" class="form-label mt-1">
                              Conversion Title
                              </label>
                            </div>
                            <div class="col-12 col-md-6 col-lg-8">
                              <input class="form-control py-1" id="row-conv-wsi-${e.path}-title"
                              required value="Testing" placeholder="Conversion title (Leave empty to use the same WSI name)">
                            </div>
                          </div>
                          <div class="row mb-2">
                            <div class="col-12 col-md-6 col-lg-4">
                              <label for="row-conv-wsi-${e.path}-size" class="form-label mt-1">
                              Tile Size (Width = Height)
                              </label>
                            </div>
                            <div class="col-12 col-md-6 col-lg-8">
                              <input class="form-control py-1" id="row-conv-wsi-${e.path}-size"
                              required value="100" placeholder="Tile size (Width = Height)">
                            </div>
                          </div>
                          <div class="row">
                            <div class="col-12 col-md-6 col-lg-4">
                            <label for="row-conv-wsi-${e.path}-overlap" class="form-label mt-1">
                              Overlap
                              </label>
                            </div>
                            <div class="col-12 col-md-6 col-lg-8">
                              <input class="form-control py-1" id="row-conv-wsi-${e.path}-overlap"
                              required value="0" placeholder="Overlap">
                            </div>
                          </div>
                        </div>
                        <div class="modal-footer">
                          <button type="button" id="row-conv-wsi-${e.path}-submit"
                          class="btn btn-primary btn-sm">Convert</button>
                          <button type="button" class="btn btn-secondary btn-sm" data-bs-dismiss="modal">Close</button>
                        </div>
                      </div>
                    </div>
                  </div>`;$("body").append(o)}),$("#wsiCountFiles").html(`<span class="text-white w-auto bg-secondary badge">
            Showing ${n} from the ${i.result.count} files in this folder.</span>`)):d.append(`<tr><td colspan="8" class="text-center align-middle">
                            <h4 class="text-danger">${i.message}</h4></td></tr>`)},error:function(i){Swal.fire({icon:"error",title:"Oops...",text:i.message})},complete:function(){s.hide(),$("table").DataTable({bDestroy:!0,initComplete:function(i,e){$("table").wrap("<div style='overflow:auto; width:100%;position:relative;'></div>")}}).draw(),g()}})};p(),$("#wsi-refresh").click(function(){d.empty(),n=0,c=0,p()}),$("#loadMoreWSIItems").off().click(function(){c+=b,p()})});