library(tidyr)
library(gdata)
library(tibble)
library(smfsb)
library(dplyr)

initial_states <-  list(
  EGFR=37,
  IGFR=5,
  SOS_inactive=100,
  SOS_active=0,
  Ras_inactive=100,
  Ras_active=0,
  PI3K_inactive=100,
  PI3K_active=0,
  AKT_inactive=100,
  AKT_active=0,
  Raf_inactive=100,
  Raf_active=0,
  Mek_inactive=100,
  Mek_active=0,
  Erk_inactive=100,
  Erk_active=0
)
rates <- list(
  SOS_activation_by_EGFR=.01,
  SOS_activation_by_IGFR=.01,
  SOS_deactivation=.5,
  Ras_activation_by_SOS=.01,
  Ras_deactivation=.5,
  PI3K_activation_by_EGFR=.01,
  PI3K_activation_by_IGFR=.01,
  PI3K_activation_by_Ras=.01,
  PI3K_deactivation=.5,
  AKT_activation_by_PI3K=.01,
  AKT_deactivation=.5,
  Raf_activation_by_Ras=.01,
  Raf_deactivation_by_AKT=.01,
  Raf_deactivation_by_phosphotase=.3,
  Mek_activation_by_Raf=.05,
  Mek_deactivation=.5,
  Erk_activation_by_Mek=.05,
  Erk_deactivation=.5
)

#####################################################################################################
######################################### ODE #######################################################
#####################################################################################################
gf_ode <- function(states, rates, interventions = NULL){
  innerRates <- rates
  innerStates <- states
  innerIntervention <- interventions
  
  if(!is.null(interventions)) {
    for(int in names(interventions)){
      innerStates[[int]] <- interventions[[int]]
    }
  }
  
  transition_function <- function(t, states = innerStates, rates = innerRates, interventions = innerIntervention) {
    
    with(as.list(c(states, rates)), {
      # update states with interventions 
      if(!is.null(interventions)) {
        for(int in names(interventions)){
          states[[int]] <- interventions[[int]]
        }
      }
      dEGFR <- 0
      dIGFR <- 0
      if("SOS_inactive" %in% names(interventions) & "SOS_active" %in% names(interventions)) {
        dSOS_inactive <- 0
        dSOS_active
      }else {
        dSOS_inactive <- -SOS_activation_by_EGFR * SOS_inactive * EGFR + -SOS_activation_by_IGFR * SOS_inactive * IGFR + SOS_deactivation * SOS_active
        dSOS_active <- SOS_activation_by_EGFR * SOS_inactive * EGFR + SOS_activation_by_IGFR * SOS_inactive * IGFR + -SOS_deactivation * SOS_active
      }
      if("Ras_inactive" %in% names(interventions) & "Ras_active" %in% names(interventions)) {
        dRas_inactive <- 0
        dRas_active <- 0
      }else {
        dRas_inactive <- -Ras_activation_by_SOS * Ras_inactive * SOS_active + Ras_deactivation * Ras_active
        dRas_active <- Ras_activation_by_SOS * Ras_inactive * SOS_active + -Ras_deactivation * Ras_active
      }
      if("PI3K_inactive" %in% names(interventions) & "PI3K_active" %in% names(interventions)) {
        dPI3K_inactive <- 0
        dPI3K_active <- 0
      }else {
        dPI3K_inactive <- -PI3K_activation_by_EGFR * PI3K_inactive * EGFR + -PI3K_activation_by_IGFR * PI3K_inactive * IGFR + -PI3K_activation_by_Ras * PI3K_inactive * Ras_active + PI3K_deactivation * PI3K_active
        dPI3K_active <- PI3K_activation_by_EGFR * PI3K_inactive * EGFR + PI3K_activation_by_IGFR * PI3K_inactive * IGFR + PI3K_activation_by_Ras * PI3K_inactive * Ras_active + -PI3K_deactivation * PI3K_active
      }
      if("AKT_inactive" %in% names(interventions) & "AKT_active" %in% names(interventions)) {
        dAKT_inactive <- 0
        dAKT_active <- 0
      }else {
        dAKT_inactive <- -AKT_activation_by_PI3K * AKT_inactive * PI3K_active + AKT_deactivation * AKT_active
        dAKT_active <- AKT_activation_by_PI3K * AKT_inactive * PI3K_active + -AKT_deactivation * AKT_active
      }
      if("Raf_inactive" %in% names(interventions) & "Raf_active" %in% names(interventions)) {
        dRaf_inactive <- 0
        dRaf_active <- 0
      }else {
        dRaf_inactive <- -Raf_activation_by_Ras * Raf_inactive * Ras_active + Raf_deactivation_by_phosphotase * Raf_active + Raf_deactivation_by_AKT * AKT_active * Raf_active
        dRaf_active <- Raf_activation_by_Ras * Raf_inactive * Ras_active + -Raf_deactivation_by_phosphotase * Raf_active + -Raf_deactivation_by_AKT * AKT_active * Raf_active
      }
      if("Mek_inactive" %in% names(interventions) & "Mek_active" %in% names(interventions)) {
        dMek_inactive <- 0
        dMek_active <- 0
      }else {
        dMek_inactive <- -Mek_activation_by_Raf * Mek_inactive * Raf_active + Mek_deactivation * Mek_active
        dMek_active <- Mek_activation_by_Raf * Mek_inactive * Raf_active - Mek_deactivation * Mek_active
      }
      if("Erk_inactive" %in% names(interventions) & "Erk_active" %in% names(interventions)) {
        dErk_inactive <- 0
        dErk_active <- 0
      }else {
        dErk_inactive <- -Erk_activation_by_Mek * Erk_inactive * Mek_active + Erk_deactivation * Erk_active
        dErk_active <- Erk_activation_by_Mek * Erk_inactive * Mek_active - Erk_deactivation * Erk_active
      }
      list(c(dEGFR, dIGFR, dSOS_inactive, dSOS_active, dRas_inactive, dRas_active, dPI3K_inactive, dPI3K_active, dAKT_inactive, dAKT_active, dRaf_inactive, dRaf_active, dMek_inactive, dMek_active, dErk_inactive, dErk_active))
    })
  }
  attr(transition_function, 'rates') <- rates
  return(transition_function)
}

#############
ode_sim <- function(transition_function, initial_states, times, interventions = NULL){
  if(!is.null(interventions)) {
    for(int in names(interventions)){
      initial_states[[int]] <- interventions[[int]]
    }
  }
  
  initial_states <- structure(as.numeric(initial_states), names = names(initial_states))
  rates <- attr(transition_function, 'rates')
  rates <- structure(as.numeric(rates), names = names(rates))
  as_tibble(
    as.data.frame(
      deSolve::ode(
        y = initial_states,
        times = times,
        func = transition_function,
        parms = rates
      )
    )
  )
}


### Intervene on Ras.  we only need one row from ODE
times <- seq(0, 1, by = .1)
faster_rates <- lapply(rates, `*`, 20)
stoc_transition_func <- gf_ode(initial_states, faster_rates,interventions = list(Ras_inactive=70, Ras_active=30))
int_data <- ode_sim(stoc_transition_func, initial_states, times,interventions = list(Ras_inactive=70, Ras_active=30))
igf_ode_Ras30 <- int_data[nrow(int_data),]
#write.csv(igf_ode_Ras30,"igf_ode_Ras30.csv")

#ODE observational data
stoc_transition_func <- gf_ode(initial_states, faster_rates,interventions = NULL)
obs_data <- ode_sim(stoc_transition_func, initial_states, times,interventions = NULL)
igf_ode_obs <- obs_data[nrow(obs_data),]
#write.csv(igf_ode_obs,"igf_ode_obs.csv")


#####################################################################################################
######################################### SDE #######################################################
#####################################################################################################
pre_file <- system.file('growth_factor_sheets/growth_factor', 'Values-Pre.csv', package="ode2scm")
post_file <- system.file('growth_factor_sheets/growth_factor', 'Values-Post.csv', package="ode2scm")
PRE <- as.matrix(read.csv(pre_file, header = TRUE))
POST <- as.matrix(read.csv(post_file, header = TRUE))

gf_sde <- function(states, rates, interventions = NULL){
  sde <- list()
  
  sde$Pre <- PRE
  sde$Post <- POST
  
  innerIntervention <- interventions
  
  sde$h <- function(states, t, parameters=rates, interventions = innerIntervention){
    # update the initial states
    if(!is.null(interventions)) {
      for(int in names(interventions)){
        states[[int]] <- interventions[[int]]
      }
    }
    with(as.list(c(states, parameters, interventions)), {
      if(!is.null(interventions)) {
        for(int in names(interventions)){
          sde$Pre[,int] <- 0
          sde$Post[,int] <- 0
        }
      }
      out <- c(
        SOS_activation_by_EGFR * SOS_inactive * EGFR,
        SOS_activation_by_IGFR * SOS_inactive * IGFR,
        SOS_deactivation * SOS_active,
        Ras_activation_by_SOS * Ras_inactive * SOS_active,
        Ras_deactivation * Ras_active,
        PI3K_activation_by_EGFR * PI3K_inactive * EGFR,
        PI3K_activation_by_IGFR * PI3K_inactive * IGFR,
        PI3K_activation_by_Ras * PI3K_inactive * Ras_active,
        PI3K_deactivation * PI3K_active,
        AKT_activation_by_PI3K * AKT_inactive * PI3K_active,
        AKT_deactivation * AKT_active,
        Raf_activation_by_Ras * Raf_inactive * Ras_active,
        Raf_deactivation_by_phosphotase * Raf_active,
        Raf_deactivation_by_AKT * AKT_active * Raf_active,
        Mek_activation_by_Raf * Mek_inactive * Raf_active,
        Mek_deactivation * Mek_active,
        Erk_activation_by_Mek * Erk_inactive * Mek_active,
        Erk_deactivation * Erk_active
      )
      names(out) <- c("SOS_inactive_to_SOS_active","SOS_inactive_to_SOS_active","SOS_active_to_SOS_inactive",
                       "Ras_inactive_to_Ras_active", "Ras_active_to_Ras_inactive",
                      "PI3K_inactive_to_PI3K_active","PI3K_inactive_to_PI3K_active","PI3K_inactive_to_PI3K_active","PI3K_active_to_PI3K_inactive",
                      "AKT_inactive_to_AKT_active","AKT_active_to_AKT_inactive",
                      "Raf_inactive_to_Raf_active", "Raf_active_to_Raf_inactive", "Raf_active_to_Raf_inactive",
                      "Mek_inactive_to_Mek_active", "Mek_active_to_Mek_inactive",
                      "Erk_inactive_to_Erk_active", "Erk_active_to_Erk_inactive")
      if(!is.null(interventions)) {
        for(int in names(interventions)){
          out[which(grepl(int, names(out), ignore.case = TRUE) == TRUE)] <- 0
        }
      }
      out <- unname(out)
      
      return(out)
    })
  }
  transition_function <- StepGillespie(sde)
  return(transition_function)
}

sde_sim <- function(transition_function, initial_states, times, interventions = NULL) {
  if(!is.null(interventions)) {
    for(int in names(interventions)){
      initial_states[[int]] <- interventions[[int]]
    }
  }
  initial_states <- structure(as.numeric(initial_states), names = names(initial_states))
  t_delta <- times[2] - times[1]
  out <- as_tibble(
    smfsb::simTs(initial_states, times[1], times[length(times)], t_delta, transition_function)
  )
  out$time <- times[0:(length(times)-1)]
  out <- out[, c('time', setdiff(names(out), 'time'))]
  return(out)
}

### Intervene on Raf
# times <- seq(0, 1, by = .1)
# #interventions <- NULL
# interventions <- list(Raf_inactive=80, Raf_active=20)
# faster_rates <- lapply(rates, `*`, 20)
# stoc_transition_func <- gf_sde(initial_states, faster_rates,interventions)
# sde_out <- sde_sim(stoc_transition_func, initial_states, times,interventions)


# create observational and interventional data from SDE when number of phosphorylated Ras = 30
# We can do the same thing with intervening on Mek and fixing number of phosphorylated Mek = 40

observational_data <- data.frame("time" = 0, "EGFR" = 0, "IGFR" = 0, "SOS_inactive" = 0, "SOS_active" = 0,
                 "Ras_inactive" = 0, "Ras_active" = 0,"PI3K_inactive" = 0, "PI3K_active" = 0,
                 "AKT_inactive" = 0, "AKT_active" = 0,"Raf_inactive" = 0, "Raf_active" = 0,
                 "Mek_inactive" = 0,"Mek_active" = 0, "Erk_inactive" = 0, "Erk_active" = 0)
intervention_data <- observational_data

for (i in 1:5000) {
  set.seed(i)
  print(i)
  times <- seq(0, 1.1, by = .1)
  faster_rates <- lapply(rates, `*`, 20)
  
  stoc_transition_func <- gf_sde(initial_states, faster_rates,interventions = NULL)
  sde_out <- sde_sim(stoc_transition_func, initial_states, times,interventions = NULL)
  observational_data <- rbind(observational_data,sde_out[nrow(sde_out),])
  
  stoc_transition_func <- gf_sde(initial_states, faster_rates,interventions = list(Ras_inactive=70, Ras_active=30))
  sde_out <- sde_sim(stoc_transition_func, initial_states, times,interventions = list(Ras_inactive=70, Ras_active=30))
  intervention_data <- rbind(intervention_data,sde_out[nrow(sde_out),])
}

setwd("/Users/sarataheri/GitHub/ode2scm/data")
#saveRDS(intervention_data,"intervention_igf100.RData") 

od <- readRDS("observational_igf100.RData")
observational_data <- od
observational_data <- observational_data[,c("SOS_active","Ras_active","PI3K_active","AKT_active","Raf_active","Mek_active","Erk_active")]
colnames(observational_data) <- c("SOS","Ras","PI3K","AKT","Raf","Mek","Erk")
observational_data <- observational_data[-1,]
rownames(observational_data) <- seq(1:5000)

id <- readRDS("intervention_igf100.RData")
intervention_data <- id
intervention_data <- intervention_data[,c("SOS_active","Ras_active","PI3K_active","AKT_active","Raf_active","Mek_active","Erk_active")]
colnames(intervention_data) <- c("SOS","Ras","PI3K","AKT","Raf","Mek","Erk")
intervention_data <- intervention_data[-1,]
rownames(intervention_data) <- seq(1:5000)


write.csv(observational_data,"/Users/sarataheri/GitHub/ode2scm/data/observational_igf.csv")
write.csv(intervention_data,"/Users/sarataheri/GitHub/ode2scm/data/intervention_igf.csv")

#causal effect of do(Ras = 30) on Erk
library(ggplot2)
CE_on_Erk <- observational_data$Erk - intervention_data$Erk
CE_on_Erk_df <- data.frame("CE_on_Erk" = CE_on_Erk)
ggplot(CE_on_Erk_df, aes(x = CE_on_Erk)) + geom_histogram()
