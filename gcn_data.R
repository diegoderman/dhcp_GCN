
devtools::load_all("~/dhcpr")
devtools::install_github(repo = "mandymejia/ciftiTools", ref = "8.0")
#library("fmriTools")
ciftiTools.setOption('wb_path', wb_path())

baby_ICA = "/data/dderman/baby_ICA"
cifti_fname = paste0(baby_ICA, "/groupICA_rev/results/dhcp_s-24_c-20_k-4/melodic_IC.dscalar.nii")
VSC = 32492 # number of vertices of dhcp atlas with subcort greyordinates.
Q = 20 # number of components in groupICA 
subject_list = read.csv(paste0(baby_ICA, "/3rd_release/results/db/r3d_dvars.csv"))
subjects = subject_list$participant_id
sessions = subject_list$session_id
outdir = paste0(baby_ICA, "/gcn/data")

for (i in 7:10){
  
  subid = subjects[i]
  sesid = sessions[i]

  ttest_cifti = readRDS(paste0("/data/dderman/baby_ICA/template_ICA/sub-",
                               subid, "/sub-", subid, "_ses-", sesid, "_desc-rev4_lnorm-0_tis-3_dris-3.RDS"))
  ttest = as.matrix(ttest_cifti$subjICmean / ttest_cifti$subjICse)
  ttest_subcortical_left = matrix(nrow = VSC, ncol=Q) 
  ttest_subcortical_left[ttest_cifti$subjICmean$meta$cortex$medial_wall_mask$left] = ttest[1:(V()/2),]
  ttest_subcortical_right = matrix(nrow = VSC, ncol=Q) # number of vertices of dhcp atlas with subcort greyordinates.
  ttest_subcortical_right[ttest_cifti$subjICmean$meta$cortex$medial_wall_mask$right] = ttest[(V()/2 + 1):V(),]
  
  ttest_subcortical = rbind(ttest_subcortical_left, ttest_subcortical_right)
  ttest_subcortical[is.na(ttest_subcortical)] = 0
  
  write.table(ttest_subcortical, file=paste0(outdir, "/sub-", subid, ".txt"), row.names=FALSE, col.names=FALSE, quote=FALSE, sep = ",")

}
