"""
    sqp_qpsub_construct()

- construct qpsub from mod.sqp_sol
"""

function sqp_qpsub_construct(
    mod1::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    )

    @inbounds begin 
        pi_14 = -ones(4,data.nline) #set multiplier for the hessian evaluation 14h 14i 14j 14k
        is_Hs_sym = zeros(data.nline) #is Hs symmetric
        is_Hs_PSD = zeros(data.nline) #is Hs positive semidefinite 


        #gen bound
        data.pgmax .-= value.(pg)  
        data.pgmin .-= value.(pg)
        data.qgmax .-= value.(qg)
        data.qgmin .-= value.(qg)

        #new cost coeff
        data.c1 .+= 2*data.c2.*value.(pg)
        
        #w theta bound
        for l = 1: data.nline
            mod1.ls[l,1] = -2*data.FrVmBound[2*l]*data.ToVmBound[2*l] #wijR lb
            mod1.us[l,1] = 2*data.FrVmBound[2*l]*data.ToVmBound[2*l] #wijR ub
            mod1.ls[l,2] = -2*data.FrVmBound[2*l]*data.ToVmBound[2*l] #wijI lb
            mod1.us[l,2] = 2*data.FrVmBound[2*l]*data.ToVmBound[2*l] #wijI ub

            mod1.ls[l,3] = data.FrVmBound[1+2*(l-1)]^2 - value.(line_var)[3,l] #wi lb
            mod1.us[l,3] = data.FrVmBound[2*l]^2 - value.(line_var)[3,l] #wi ub
            mod1.ls[l,4] = data.ToVmBound[1+2*(l-1)]^2 - value.(line_var)[4,l] #wj lb
            mod1.us[l,4] = data.ToVmBound[2*l]^2 - value.(line_var)[4,l] #wj ub

            mod1.ls[l,5] = data.FrVaBound[1+2*(l-1)] - value.(line_var)[5,l] #ti lb
            mod1.us[l,5] = data.FrVaBound[2*l] - value.(line_var)[5,l] #ti ub
            mod1.ls[l,6] = data.ToVaBound[1+2*(l-1)] - value.(line_var)[6,l] #tj lb
            mod1.us[l,6] = data.ToVaBound[2*l] - value.(line_var)[6,l] #tj ub
        end

        for b = 1:data.nbus
            data.Pd[b] = data.baseMVA * (data.Pd[b]/data.baseMVA - (value.(pgb)[b] - value.(pft)[b] - value.(ptf)[b] - data.YshR[b]*value.(bus_w)[b]))
            data.Qd[b] = data.baseMVA * (data.Qd[b]/data.baseMVA - (value.(qgb)[b] - value.(qft)[b] - value.(qtf)[b] + data.YshI[b]*value.(bus_w)[b]))
        end

        for l = 1: data.nline
            #Hs:(6,6) #w_ijR, w_ijI, w_i, w_j, theta_i, theta_j
            
            Hs = zeros(6,6)

            Hs_14h = zeros(6,6)
            Hs_14h[1,1] = 2*pi_14[1,l]
            Hs_14h[2,2] = 2*pi_14[1,l] 
            Hs_14h[3,4] = -pi_14[1,l] 
            Hs_14h[4,3] = -pi_14[1,l] 

            Hs_14i = zeros(6,6)
            cons_1 = pi_14[2,l]*cos(value.(line_var)[5,l] - value.(line_var)[6,l])
            cons_2 = pi_14[2,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l])
            cons_3 = pi_14[2,l]*(-value.(line_var)[1,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l]) +  value.(line_var)[1,2]*cos(value.(line_var)[5,l] - value.(line_var)[6,l]))
        
            Hs_14i[1,5] = cons_1 #wijR theta_i
            Hs_14i[5,1] = cons_1 #wijR theta_i
            Hs_14i[1,6] = -cons_1 #wijR theta_j
            Hs_14i[6,1] = -cons_1 #wijR theta_j 

            Hs_14i[2,5] = cons_2 #wijR theta_i
            Hs_14i[5,2] = cons_2 #wijR theta_i
            Hs_14i[2,6] = -cons_2 #wijR theta_j
            Hs_14i[6,2] = -cons_2 #wijR theta_j 

            Hs_14i[5,5] = cons_3 #thetai thetai
            Hs_14i[6,6] = cons_3 #thetaj thetaj
            Hs_14i[5,6] = -cons_3 #thetai thetaj
            Hs_14i[6,5] = -cons_3 #thetaj thetai 
            
            supY = [data.YftR[l] data.YftI[l] data.YffR[l] 0 0 0;
                -data.YftI[l] data.YftR[l] -data.YffI[l] 0 0 0;
                data.YtfR[l] -data.YtfI[l] 0 data.YttR[l] 0 0;
                -data.YtfI[l] -data.YtfR[l] 0 -data.YttI[l] 0 0]
            Hs_14j = -2*pi_14[3,l]*(supY[1,:]*transpose(supY[1,:]) + supY[2,:]*transpose(supY[2,:]) )
            Hs_14k = -2*pi_14[4,l]*(supY[3,:]*transpose(supY[3,:]) + supY[4,:]*transpose(supY[4,:]) )
            Hs .= Hs_14h + Hs_14i + Hs_14j + Hs_14k + UniformScaling(4)#with multiplier pi_14 #! fix to psd for now
            mod1.Hs[6*(l-1)+1:6*l,1:6] .= Hs

            is_Hs_sym[l] = maximum(abs.(Hs - transpose(Hs)))
            eival, eivec = eigen(Hs)
            is_Hs_PSD[l] = minimum(eival)

            #inherit structure of Linear Constraint (overleaf): ignore 1h and 1i with zero assignment in ipopt benchmark
            LH_1h = [2*value.(line_var[1,l]), 2*value.(line_var[2,l]), -value.(line_var[4,l]), -value.(line_var[3,l])] #LH * x = RH
            mod1.LH_1h[l,:] .= LH_1h
            RH_1h = -(value.(line_var)[1,l])^2 - (value.(line_var)[2,l])^2 + value.(line_var)[3,l]*value.(line_var)[4,l] 
            mod1.RH_1h[l] = RH_1h

            LH_1i = [sin(value.(line_var)[5,l] - value.(line_var)[6,l]), -cos(value.(line_var)[5,l] - value.(line_var)[6,l]), 
            value.(line_var)[1,l]*cos(value.(line_var)[5,l] - value.(line_var)[6,l]) +  value.(line_var)[2,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l]),
            -value.(line_var)[1,l]*cos(value.(line_var)[5,l] - value.(line_var)[6,l]) -  value.(line_var)[2,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l])] #Lf * x = RH
            mod1.LH_1i[l,:] .= LH_1i
            RH_1i = -value.(line_var)[1,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l])  +  value.(line_var)[2,l]*cos(value.(line_var)[5,l] - value.(line_var)[6,l])
            mod1.RH_1i[l] = RH_1i

            #inherit structure line limit constraint (overleaf)
            LH_1j = [2*value.(line_fl)[1,l], 2*value.(line_fl)[2,l]] #rand(2)
            mod1.LH_1j[l,:] .= LH_1j
            RH_1j = -((value.(line_fl)[1,l])^2 + (value.(line_fl)[2,l])^2 - data.rateA[l]) 
            mod1.RH_1j[l] = RH_1j

            LH_1k = [2*value.(line_fl[3,l]), 2*value.(line_fl[4,l])] #zeros(2) #rand(2)
            mod1.LH_1k[l,:] .= LH_1k
            RH_1k = -((value.(line_fl)[3,l])^2 + (value.(line_fl)[4,l])^2 - data.rateA[l]) 
            mod1.RH_1k[l] = RH_1k
        end
    end #inbound 

    return
end