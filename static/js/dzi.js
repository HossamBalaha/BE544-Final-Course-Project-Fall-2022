$(document).ready(function(){let z=0,E=10,M=0,v="None",h=$("#dzi-body"),I=$("#loading-blocker");function _(){$('button[id^="row-delete-dzi-"]').off().click(function(e){Swal.fire({title:"Are you sure?",text:"You won't be able to revert this!",icon:"warning",showCancelButton:!0,confirmButtonColor:"#3085d6",cancelButtonColor:"#d33",confirmButtonText:"Yes, delete it!"}).then(o=>{if(o.isConfirmed){I.show();let i=$(this);$.ajax({url:"/api/dzi/delete",type:"POST",data:{item:i.attr("id"),csrf_token:CSRF_TOKEN},success:function(n){n.is_success?Swal.fire({icon:"success",title:"Deleted",text:n.message}):Swal.fire({icon:"error",title:"Oops...",text:n.message})},error:function(n){Swal.fire({icon:"error",title:"Oops...",text:n.message})},complete:function(){I.hide(),$("#dzi-refresh").click()}})}})}),$('button[id^="row-view-dzi-"]').off().click(function(e){let o=$(this),i=o.attr("id").replace("row-view-dzi-","row-modal-dzi-"),n=$(document.getElementById(i));n.modal("show");let p=n.find(".modal-body");n.off("shown.bs.modal").on("shown.bs.modal",function(){p.empty(),p.append(`<div class="offcanvas offcanvas-start" tabindex="-1" id="dziNotesCanvas" aria-labelledby="dziNotesCanvasLabel">
  <div class="offcanvas-header">
    <h5 class="offcanvas-title" id="dziNotesCanvasLabel">
    Notes
</h5>
    <button type="button" class="btn-close text-reset" data-bs-dismiss="offcanvas" aria-label="Close"></button>
  </div>
  <div class="offcanvas-body">
                      <ul class="m-0 p-0 ps-1">
                        <li>Use the mouse wheel to zoom in and out.</li>
                        <li>Click and drag to pan around the image.</li>
                        <li>Hover on the image to overlay it.</li>
                        <li>After hovering the tile, you can press "Space" from the keyboard to open the tile in a new tab.</li>
                        <li>After hovering the tile, you can press "Q" from the keyboard to start image processing.</li>
                        <li>Press on "V" from the keyboard to select a segmentation run (i.e., segmenter) to apply the 
                        segmentation on the tile. The default is "No Segmentation".</li>
                        <li>After hovering the tile, you can press "G" from the keyboard to segment the selected tile.</li>
                      </ul>
  </div>
</div>
<div id="dzi-view-spinner" class="text-center"><div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span></div></div>
                    <div class="text-start" id="openseadragon-${i.replace("row-modal-dzi-","")}-label"></div>
                    <div class="text-center openseadragon"
                      id="openseadragon-${i.replace("row-modal-dzi-","")}"></div>
                  `);let t=OpenSeadragon({id:`openseadragon-${i.replace("row-modal-dzi-","")}`,prefixUrl:"https://cdn.jsdelivr.net/npm/openseadragon@3.1/build/openseadragon/images/",tileSources:URL_FOR_DZI+i.replace("row-modal-dzi-",""),degrees:0,showRotationControl:!0,gestureSettingsTouch:{pinchRotate:!0},gestureSettingsMouse:{clickToZoom:!1},debugMode:!1});t.addHandler("open-failed",function(Z){p.empty(),p.append('<div class="text-center text-danger fs-5 font-weight-bold">Error loading DZI</div>')}),t.addHandler("open",function(Z){p.find("#dzi-view-spinner").remove();let C=[],d=null;t.addHandler("tile-drawn",function(r){let l=r.tile,m=r.tiledImage,s=l.level,c=l.position,T=c.y,f=c.x,k=l.url;C.push({x:l.bounds.x,y:l.bounds.y,width:l.bounds.width,height:l.bounds.height,level:s,url:k})});let O=-1,N=0;new OpenSeadragon.MouseTracker({element:t.canvas,keyDownHandler:function(r){if(r.keyCode===86){let l=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-segmenter",m=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-segmenter-apply",s=$(document.getElementById(l)),c=$(document.getElementById(m));v="None",c.off().click(function(T){v=s.find("select").val().trim(),s.modal("hide")}),s.modal("show")}if(d!=null){if(r.keyCode===32)r.preventDefaultAction=!0,d&&window.open(d.url,"_blank");else if(r.keyCode===71)v!=="None"&&$.ajax({url:"/api/tile/segment",type:"POST",data:{tile:d.url,segmenter:v,csrf_token:CSRF_TOKEN},success:function(l){if(l.is_success){let m=document.createElement("div");m.style.background="rgba(255, 0, 0, 0.05)",m.style.border="1px solid green";let s=document.createElement("img");s.src=`data:image/jpeg;base64,${l.result}`,s.style.width="100%",s.style.height="100%",m.appendChild(s),t.clearOverlays(),t.addOverlay(m,new OpenSeadragon.Rect(d.x,d.y,d.width,d.height))}else Swal.fire({icon:"error",title:"Oops...",text:l.message})},error:function(l){Swal.fire({icon:"error",title:"Oops...",text:l.message})}});else if(r.keyCode===81&&(r.preventDefaultAction=!0,d)){let l=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-dip";$(document.getElementById(l)).modal("show");let s=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-dip-tile";$(document.getElementById(s)).attr("src",d.url);let T=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-dip-ptile",f=$(document.getElementById(T));f.attr("src",d.url);let k=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-dip-apply",w=$(document.getElementById(k)),y=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-dip-slider",g=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-dip-slider-span",a=$(document.getElementById(y)),A=$(document.getElementById(g)),j=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-dip-medblur",V=o.attr("id").replace("row-view-dzi-","row-modal-dzi-")+"-dip-medblur-span",B=$(document.getElementById(j)),P=$(document.getElementById(V));w.off().click(function(R){let u=$(this);u.attr("disabled",!0).addClass("disabled"),$.ajax({url:"/api/tile/dip",type:"POST",data:{tile:d.url,threshold:a.slider("option","value"),medblur:B.slider("option","value"),csrf_token:CSRF_TOKEN},success:function(x){x.is_success?f.attr("src",`data:image/jpeg;base64,${x.result}`):Swal.fire({icon:"error",title:"Oops...",text:x.message})},error:function(x){Swal.fire({icon:"error",title:"Oops...",text:x.message})},complete:function(){u.removeAttr("disabled").removeClass("disabled"),u.find(".fa-spinner").addClass("d-none")}})}),a.slider({value:N,min:1,max:255,step:1,slide:function(R,u){A.text(u.value),N=u.value}}),A.text(a.slider("option","value")),B.slider({value:O,min:-1,max:15,step:2,slide:function(R,u){P.text(u.value),O=u.value}}),P.text(B.slider("option","value"))}}},moveHandler:function(r){let l=t.source.dimensions.x,m=t.source.dimensions.y,s=r.position,c=t.viewport.pointFromPixel(s),T=t.viewport.viewportToImageCoordinates(c),f=t.viewport.getZoom(!0),k=t.viewport.viewportToImageZoom(f),w=-1,y=-1;d=null;for(let g=0;g<C.length;g++){let a=C[g];c.x>=a.x&&c.x<=a.x+a.width&&c.y>=a.y&&c.y<=a.y+a.height&&a.level>y&&(y=a.level,w=g)}if(w>=0&&y>=0){d=C[w];let g=d.url,a=document.createElement("div");a.style.background="rgba(255, 0, 0, 0.05)",a.style.border="1px solid green",t.clearOverlays(),t.addOverlay(a,new OpenSeadragon.Rect(d.x,d.y,d.width,d.height))}}}).setTracking(!0)})}),n.off("hidden.bs.modal").on("hidden.bs.modal",function(){p.empty(),v="None"})})}let D=()=>{$("table").DataTable().destroy(),I.show(),h.append(`<tr id="tr-loader"><td colspan="8" class="text-center align-middle">
          <div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></td></tr>`),$.ajax({url:"/api/dzi/browse",type:"GET",data:{from:z,step:E},success:function(b){if(z===0&&h.empty(),$("#tr-loader").remove(),b.is_success){let S="";b.result.elements.forEach((e,o)=>{let i=$('<tr id="row"></tr>');i.append(`<td class="text-center align-middle">${++M}</td>`),i.append(`<td class="text-center align-middle">${e.name}</td>`),i.append(`<td class="text-center align-middle">${e.modifiedAt}</td>`),i.append(`<td class="text-center align-middle">${e.createdAt}</td>`),i.append(`<td class="text-center align-middle">${e.accessedAt}</td>`),i.append(`<td class="text-center align-middle">${e.segmentations.length}</td>`);let n="";for(let t=0;t<e.segmentations.length;t++)n+=`<option value="${e.segmentations[t][0]}">
                ${e.segmentations[t][1]}
              </option>`;let p="";p=`
                  <button class="btn btn-sm btn-primary" type="button" title="View DZI"
                    id="row-view-dzi-${e.path}">
                    <i class="fa fa-eye"></i>
                  </button>
                  <button class="btn btn-sm btn-danger" title="Delete DZI" id="row-delete-dzi-${e.path}">
                    <i class="fa fa-trash"></i>
                  </button>`,S+=`<div class="modal fade" id="row-modal-dzi-${e.path}" tabindex="-1"
                    aria-labelledby="row-modal-dzi-${e.path}-label" aria-hidden="true">
                    <div class="modal-dialog modal-full-screen mx-auto modal-scrollable modal-dialog-centered" role="document">
                      <div class="modal-content">
                        <div class="modal-header text-center">
                          <h1 class="modal-title fs-5 text-center" id="row-modal-dzi-${e.path}-label">
                            Displaying DZI: <b>${e.name}</b>
                            <button class="btn btn-info btn-sm mx-2" type="button" data-bs-toggle="offcanvas" 
                              data-bs-target="#dziNotesCanvas" aria-controls="dziNotesCanvas">
                              Open Notes
                            </button>
                          </h1>
                          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body p-1">
                        </div>
                        <div class="modal-footer">
                          <button type="button" class="btn btn-secondary btn-sm" data-bs-dismiss="modal">Close</button>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div class="modal fade" id="row-modal-dzi-${e.path}-dip" tabindex="-1"
                    aria-labelledby="row-modal-dzi-${e.path}-dip-label" aria-hidden="true">
                    <div class="modal-dialog mx-auto modal-dialog-centered" role="document">
                      <div class="modal-content bg-danger">
                        <div class="modal-header text-center">
                          <h1 class="modal-title fs-5 text-center" id="row-modal-dzi-${e.path}-dip-label">
                            Working on the Selected Tile.
                          </h1>
                          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                                <div class="row">
                                <div class="col mx-auto">
                                <p class="text-center font-weight-bold m-0 p-0 mb-1">Tile</p>
                                  <div class="text-center">
                                    <img src="" alt="Tile"
                                      class="shadow shadow-lg" id="row-modal-dzi-${e.path}-dip-tile">
                                  </div>
</div>
                                <div class="col mx-auto">
                                <p class="text-center font-weight-bold m-0 p-0 mb-1">Processed Tile</p>
                                  <div class="text-center">
                                    <img src="" alt="Processed Tile"
                                      class="shadow shadow-lg" id="row-modal-dzi-${e.path}-dip-ptile">
                                  </div>
</div>
</div>
                            <div class="row">
                            <div class="col-12">
                                  <p class="text-center font-weight-bold m-0 p-0 mt-2 h4">Controls</p>
                                  <p class="font-weight-bold mb-1">
                                    Threshold Value:
                                    <span id="row-modal-dzi-${e.path}-dip-slider-span" class="badge bg-success"></span>
                                  </p>
                                  <div id="row-modal-dzi-${e.path}-dip-slider"></div>
                                  <p class="font-weight-bold mb-1 mt-2">
                                    Median Bluring Value (-1 for no Bluring):
                                    <span id="row-modal-dzi-${e.path}-dip-medblur-span" class="badge bg-success"></span>
                                  </p>
                                  <div id="row-modal-dzi-${e.path}-dip-medblur"></div>
                                  <div class="text-center mt-2">
                                    <button class="btn btn-sm btn-primary bg-primary" 
                                      id="row-modal-dzi-${e.path}-dip-apply"
                                      type="button" title="Apply Changes">
                                      Apply Visually
                                    </button>
                                  </div>
                            </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                          <button type="button" class="btn btn-secondary btn-sm" 
                            data-bs-dismiss="modal">Close</button>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div class="modal fade" id="row-modal-dzi-${e.path}-segmenter" tabindex="-1"
                    aria-labelledby="row-modal-dzi-${e.path}-segmenter-label" aria-hidden="true">
                    <div class="modal-dialog mx-auto modal-dialog-centered" role="document">
                      <div class="modal-content bg-danger">
                        <div class="modal-header text-center">
                          <h1 class="modal-title fs-5 text-center" id="row-modal-dzi-${e.path}-segmenter-label">
                            Select a Segmentation Run
                          </h1>
                          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                      <select class="form-select d-block w-100" required>
                        <option value="None" selected>No Segmentation</option>
                        ${n}
                      </select>
                    <div class="text-center mt-2">
                                    <button class="btn btn-sm btn-primary bg-primary" 
                                      id="row-modal-dzi-${e.path}-segmenter-apply"
                                      type="button" title="Apply Segmentation Run">
                                      Apply Segmentation Run
                                    </button>
                                  </div>
                        </div>
                        <div class="modal-footer">
                          <button type="button" class="btn btn-secondary btn-sm" 
                            data-bs-dismiss="modal">Close</button>
                        </div>
                      </div>
                    </div>
                  </div>`,i.append(`<td class="text-center align-middle"><div>${p}</div></td>`),h.append(i)}),$("body").append(S),$("#dziCountFiles").html(`<span class="text-white w-auto bg-secondary badge">
            Showing ${M} from the ${b.result.count} files in this folder.</span>`)}else h.append(`<tr><td colspan="8" class="text-center align-middle">
                <h4 class="text-danger">${b.message}</h4></td></tr>`)},error:function(b){Swal.fire({icon:"error",title:"Oops...",text:b.message})},complete:function(){I.hide(),$("table").DataTable({bDestroy:!0,initComplete:function(b,S){$("table").wrap("<div style='overflow:auto; width:100%;position:relative;'></div>")}}).draw(),_()}})};D(),$("#dzi-refresh").click(function(){h.empty(),M=0,z=0,D()}),$("#loadMoreDZIItems").off().click(function(){z+=E,D()})});