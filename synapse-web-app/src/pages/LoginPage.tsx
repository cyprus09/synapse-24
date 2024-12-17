import { useState } from "react";
import loginImage from "../assets/loginImage.png";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { useNavigate } from "react-router-dom";

const LoginPage = () => {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = e => {
    e.preventDefault();
    setLoading(true);
    navigate("/dashboard");
    setLoading(false);
  };

  return (
    <div className="flex flex-col lg:flex-row w-full min-h-screen">
      <div className="relative w-full lg:w-1/2 h-64 lg:h-screen">
        <img src={loginImage} className="absolute inset-0 w-full h-full object-cover" alt="auth-image" />
        <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
          <div className="text-center text-white p-6 md:p-8">
            <h1 className="text-3xl md:text-4xl font-bold mb-4">Join Now !!!</h1>
          </div>
        </div>
      </div>

      <div className="w-full lg:w-1/2 bg-background flex flex-col justify-center p-6 md:p-12 lg:p-16">
        <h1 className="text-xl text-foreground font-semibold mb-8 text-center">App Name Goes Here</h1>

        <Card className="w-full max-w-md mx-auto">
          <CardHeader>
            <CardTitle className="text-lg font-bold">Login</CardTitle>
            <CardDescription className="text-muted-foreground">
              Welcome Back! Please enter your details.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-6" onSubmit={handleSubmit}>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" type="email" placeholder="Enter your email" required />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <Input id="password" type="password" placeholder="Enter your password" required />
                </div>
              </div>
              <div className="space-y-4">
                <Button type="submit" className="w-full" disabled={loading}>
                  {loading ? "Loading..." : "Log In"}
                </Button>
                <Button className="w-full bg-white" variant="outline" onClick={() => navigate("/register")}>
                  Register
                </Button>
                <div className="flex items-center my-8">
                  <Separator className="flex-1" />
                  <span className="px-4 text-muted-foreground">or</span>
                  <Separator className="flex-1" />
                </div>
                <Button
                  className="w-full bg-white border border-gray-300 text-gray-700 hover:bg-gray-100"
                  variant="outline"
                  disabled={loading}
                >
                  <svg
                    className="mr-2 h-4 w-4"
                    aria-hidden="true"
                    focusable="false"
                    data-prefix="fab"
                    data-icon="google"
                    role="img"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 488 512"
                  >
                    <path
                      fill="currentColor"
                      d="M488 261.8C488 403.3 391.1 504 248 504 110.8 504 0 393.2 0 256S110.8 8 248 8c66.8 0 123 24.5 166.3 64.9l-67.5 64.9C258.5 52.6 94.3 116.6 94.3 256c0 86.5 69.1 156.6 153.7 156.6 98.2 0 135-70.4 140.8-106.9H248v-85.3h236.1c2.3 12.7 3.9 24.9 3.9 41.4z"
                    ></path>
                  </svg>
                  Sign In With Google
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default LoginPage;
