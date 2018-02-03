
## Draw contour lines

library(mvtnorm)

add.contour <- function(rho){

  x.points <- seq(-3,3,length=100) 
  y.points <- x.points
  z <- matrix(0,nrow=100,ncol=100)
  mu <- c(0,0)
  sigma <- matrix(c(1,rho,rho,1),nrow=2) 

  for (i in 1:100) {
    for (j in 1:100) {
      z[i,j] <- dmvnorm(c(x.points[i],y.points[j]),
                      mean=mu,sigma=sigma)
    } 
  }
  contour(x.points,y.points,z,col="blue",add=TRUE,levels=c(0.01,0.1,0.3,0.5,0.7,0.9))
}



## Sample for bivariate normal using Choleski decomposition

rho=0.99

S=matrix(c(1,rho,rho,1),ncol=2)
R=chol(S)
N=100
sn=matrix(rnorm(2*N),nrow=2) ##Simulate iid N(0,1) variables
sim_chol=t(R)%*%sn
plot(sim_chol[1,],sim_chol[2,],col="red",xlim=c(-3.5, 3.5), ylim=c(-3.5, 3.5), xlab="Direct sampling", ylab="")
add.contour(rho)




## Metropolis random walk for bivariate normal distribution



metropolis=function(rho,N,sv,sigma){
  
  S=matrix(c(1,rho,rho,1),ncol=2)
  Si=solve(S)
  sim_rwm=matrix(NA,nrow=2,ncol=N+1)
  prop=matrix(NA,nrow=2,ncol=N+1)
  sim_rwm[,1]=sv   
  prop[,1]=sv 
  #Deterministic initial value

  a.ratio = function(xprev,xprop,Si){
    exp(-0.5*(t(xprop)%*%Si%*%xprop-t(xprev)%*%Si%*%xprev))
  } 
  #Computing acceptance ratio 
  
  acc=matrix(FALSE,nrow=1,ncol=N+1)
  acc[1]=TRUE
  
  for(i in 1:N){
    prop[,i+1]=sim_rwm[,i]+rnorm(2,sd=sigma)
    if(runif(1)<=a.ratio(xprev=sim_rwm[,i],xprop=prop[,i+1],Si=Si)){
      sim_rwm[,i+1]=prop[,i+1]
      acc[i+1]=TRUE
    }else{
      sim_rwm[,i+1]=sim_rwm[,i]
    }
  }
  
  plot(sim_rwm[1,acc],sim_rwm[2,acc],xlim=c(-5,5),ylim=c(-5,5),col="red") #accepted
  points(prop[1,!acc],prop[2,!acc],col="green")     #not accepted
  lines(sim_rwm[1,acc],sim_rwm[2,acc],col="red")
  length(acc[acc])/N
}

rho=0.99
N=200
sigma=0.1
x_start=c(5,5)

metropolis(rho,N,x_start,sigma)
add.contour(rho)






## Hybrid MC for bivariate normal distribution


hybrid=function(rho,N,sv,epsilon,steps){
  
  S=matrix(c(1,rho,rho,1),ncol=2)
  Si=solve(S)
  prop_rwm=sim_rwm=matrix(NA,nrow=2,ncol=N+1) 
  prop_rwm[,1]=sim_rwm[,1]=sv
  
  neg.log.pi=function(x,Si) return(0.5*t(x)%*%Si%*%x)  #potential energy target distribution
  grad.neg.log.pi=function(x,Si) return(Si%*%x)        #gradient
  
  acc=matrix(FALSE,nrow=1,ncol=N+1)
  acc[1]=TRUE # accepted proposals
  
  for(i in 1:N){
    
    current_x=sim_rwm[,i]
    x = current_x
    u = rnorm(length(x),0,1)  # independent standard normal (mass=1 for all particles)
  
  # Use the leapfrog method to generate proposal
    
    current_u = u
    u = u - epsilon * grad.neg.log.pi(x,Si=Si) / 2

  for (j in 1:steps){
    x = x + epsilon * u
    if (j!=steps) u = u - epsilon * grad.neg.log.pi(x,Si=Si)
  }
    u = u - epsilon * grad.neg.log.pi(x,Si=Si) / 2
  
  # End of leapfrog iteration
    
    
  current_H= neg.log.pi(current_x,Si=Si)+ sum(current_u^2) / 2  #current Hamiltonian
  proposed_H = neg.log.pi(x,Si=Si)+ sum(u^2) / 2                #proposed Hamiltonian
  prop_rwm[,i+1]=x
  
  #acceptance step
    if (runif(1) < exp(current_H-proposed_H)){
      sim_rwm[,i+1]=x
      acc[i+1]=TRUE
    } else {
      sim_rwm[,i+1]=sim_rwm[,i]
    }
  }
  
  plot(sim_rwm[1,acc],sim_rwm[2,acc],xlim=c(-5,5),ylim=c(-5,5),col="red") #accepted
  points(prop_rwm[1,!acc],prop_rwm[2,!acc],col="green")     #not accepted
  lines(sim_rwm[1,acc],sim_rwm[2,acc],col="red")
  length(acc[acc])/N
}
  
rho=0.99
N=20
sv=c(5,5)
epsilon=0.1
steps=10

hybrid(rho,N,sv,epsilon,steps)
add.contour(rho)






## Gibbs sampler for bivariate normal distribution



gibbs=function(rho,N,sv){
  
  sim_gibbs=matrix(NA,nrow=2,ncol=N+1)
  sim_gibbs[,1]=sv  ##Deterministic initial value
  for(i in 1:N){
    sim_gibbs[1,i+1]=rnorm(1,mean=rho*sim_gibbs[2,i],sd=sqrt(1-rho^2))
    sim_gibbs[2,i+1]=rnorm(1,mean=rho*sim_gibbs[1,i+1],sd=sqrt(1-rho^2))
  }
  
  plot(sim_gibbs[1,],sim_gibbs[2,],xlim=c(-5,5),ylim=c(-5,5),col="red") #accepted

  lines(sim_gibbs[1,],sim_gibbs[2,],col="red")

}

rho=0.99
N=50
x_start=c(5,5)

gibbs(rho,N,x_start)
add.contour(rho)
