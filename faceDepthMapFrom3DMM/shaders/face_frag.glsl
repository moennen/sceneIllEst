#version 410

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform float ambient;

uniform float roughness;
uniform float sheen;
uniform float subsurface;

in vec3 colour;
in vec4 position;
in vec3 normal;
in vec2 uv;

layout(location = 0) out vec4 frag_colour;
layout(location = 1) out vec4 frag_uv_depth;
layout(location = 2) out vec4 frag_normals;

vec3 applyBRDF( vec3 normal,
                vec3 pos,
                vec3 lightPos,
                vec3 lightCol,
                vec3 baseCol,
                float ao,
                float roughness,
                float subsurface,
                float metallic,
                float spec,
                vec3 specCol,
                float sheen,
                vec3 sheenCol,
                float clearcoat,
                float clearcoatGloss );

void main() 
{
   vec3 norm = normalize(normal);
   
   vec3 col = applyBRDF(norm,position.xyz,
                        lightPos, 0.5*lightColor,
                        colour,
                        1.0,roughness,subsurface,0.0,0.5,vec3(1.0,1.0,1.0),
                        sheen, vec3(1.0,1.0,1.0), 0.0, 0.0 ) + ambient * colour;

   float depth =  -position.z / 1000.0;
   frag_colour = vec4(col,1.0);
   frag_uv_depth = vec4(uv, depth, 1.0);
   frag_normals = vec4(norm, 1.0);
}


//-------------------------------------------------------------------------------------------------
// BRDF 

const float oneOverPi = 1.0 / 3.14159265358979;
float sqr(float x) {return x*x;}

// Microfacet distribution function : Towbridge-Reitz (GTR) NDF
float d_gtr(float ndoth, vec3 h, float roughness)
{
   float r2 = sqr(roughness);
   float r4=sqr(r2);
   float D=ndoth*ndoth*(r4-1.0)+1.0;
   return oneOverPi*r4/max(sqr(D),1e-8);
}

// Schlick Fresnel approximation
float f_schlick(float x)
{
   float f = clamp(1.0-x, 0.0, 1.0);
   float f2 = f*f;
   return f2*f2*f; // f^5
}

// Diffuse computation :
// following [4] the roughness is squared and the
// diffuse is renormalized to be conservative
float diff_pbs(float ndotl, float ndote, float hdotl,
               float roughness, float subsurface, float metallic)
{
   // Diffuse Fresnel component
   float fl = f_schlick(ndotl);
   float fe = f_schlick(ndote);
   float roughnessSqr=sqr(roughness);
   float energyBias = mix(0.0, 0.5, roughnessSqr);
   float energyFactor = mix(1.0, 1.0 / 1.51, roughnessSqr);
   float f90 = sqr(hdotl)*roughnessSqr;
   float fd90 = energyBias + 2.0 * f90;
   float fd = mix(1.0, fd90, fl) * mix(1.0, fd90, fe);
   // compute the subsurface approximation based on Hanrahan-Krueger (see [2])
   float fdss = mix(1.0, f90, fl) * mix(1.0, f90, fe);
   float indotle = 1.0/(ndotl+ndote);
   fdss = 1.25*(fdss*(indotle-energyBias)+energyBias);

   return oneOverPi * mix(fd,fdss,subsurface) * (1.0-metallic) * energyFactor;
}

// PBS
vec3 applyBRDF( vec3 normal,
                vec3 pos,
                vec3 lightPos,
                vec3 lightCol,
                vec3 baseCol,
                float ao,
                float roughness,
                float subsurface,
                float metallic,
                float spec,
                vec3 specCol,
                float sheen,
                vec3 sheenCol,
                float clearcoat,
                float clearcoatGloss )
{
   const float min_val = 1e-6;

   vec3 lightDir = normalize( lightPos - pos );
   vec3 viewDir = normalize( -pos );
   vec3 halfDir = normalize( lightDir + viewDir );

   float ndotl=max(dot(normal,lightDir),min_val);
   float ndoth=max(dot(normal,halfDir),min_val);
   float ndote=max(dot(normal,viewDir),min_val);
   float hdotl=max(dot(halfDir,lightDir),min_val);

   // PBS DIFFUSE
   float diffuse = diff_pbs(ndotl, ndote, hdotl, roughness, subsurface, metallic);

   // PBS SPECULAR
   // specular D (GGX with alpha=ggxRoughness^2 [2])
   float D = d_gtr( ndoth, halfDir, roughness );
   // specular G (Schlick approximation of the Smith model for GGX [3])
   float k=(roughness+1.0)*(roughness+1.0)*0.125;
   float G=(ndote/(ndote*(1.0-k)+k))*(ndotl/(ndotl*(1.0-k)+k));
   // Fresnel component
   float f= f_schlick(hdotl);
   // based-color specular [2]
   vec3 fspec=mix(specCol,vec3(1.0), f);
   // -->
   float s=1.0/max(4.0*ndote*ndotl,min_val);
   vec3 specular=vec3(s*D*G)*fspec;

    // SHEEN [2]
   vec3 sheenColor = (f*sheen*sheenCol)*(1.0-metallic);

   // CLEARCOAT [2]
   float ccD = d_gtr(ndoth,halfDir,mix(0.1,0.001,clearcoatGloss));
   float ccf = mix(0.04, 1.0, f);
   const float cck=0.125;
   float ccG=(ndote/(ndote*(1.0-cck)+cck))*(ndotl/(ndotl*(1.0-cck)+cck));
   vec3 clearcoatColor = vec3(s*ccD*ccG*ccf*clearcoat);

   // =>
   vec3 reflectedColor =  vec3(ao) * ( baseCol * diffuse + sheenColor ) + 
                          spec * (specular + clearcoatColor);

   return lightCol * ndotl * reflectedColor;
}