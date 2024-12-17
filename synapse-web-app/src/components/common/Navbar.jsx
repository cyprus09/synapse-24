import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Menu, Home } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { nanoid } from "nanoid";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "@/hooks/use-toast";

const landings = [
  {
    id: nanoid(),
    title: "Landing 01",
    route: "/project-management",
  },
  {
    id: nanoid(),
    title: "Landing 02",
    route: "/crm-landing",
  },
  {
    id: nanoid(),
    title: "Landing 03",
    route: "/ai-content-landing",
  },
];

const Navbar = () => {
  const navigate = useNavigate();
  const { signOut, user } = useAuth();

  const handleLogout = async () => {
    try {
      await signOut();
      navigate("/login");
      toast({
        title: "Logged out successfully",
        description: "You have been logged out of your account.",
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error logging out",
        description: "There was a problem logging out. Please try again.",
      });
      console.error("Error logging out:", error);
    }
  };

  return (
    <Card className="container sticky top-4 bg-card py-3 px-4 border-0 flex flex-col rounded-2xl mt-5 my-auto mx-auto">
      <div className="flex items-center justify-between gap-6">
        <Home className="text-primary cursor-pointer" size={24} />
        <ul className="hidden md:flex gap-10 text-card-foreground">
          <li className="text-primary font-medium">
            <a href="#home">Home</a>
          </li>
          <li>
            <a href="#faqs">FAQs</a>
          </li>
        </ul>

        <div className="flex items-center">
          {/* Show logout when user is logged in */}
          <Button variant="secondary" className="hidden md:block px-2" onClick={handleLogout}>
            Logout
          </Button>

          {/* Mobile menu */}
          <div className="flex md:hidden mr-2 items-center gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon">
                  <Menu className="h-5 w-5 rotate-0 scale-100" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem>
                  <a href="#home">Home</a>
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <a href="#faqs">FAQs</a>
                </DropdownMenuItem>
                {user ? (
                  <DropdownMenuItem>
                    <Button variant="secondary" className="w-full text-sm" onClick={handleLogout}>
                      Logout
                    </Button>
                  </DropdownMenuItem>
                ) : (
                  <>
                    <DropdownMenuItem>
                      <Button variant="secondary" className="w-full text-sm">
                        <Link to="/login">Login</Link>
                      </Button>
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                      <Button className="w-full text-sm">
                        <Link to="/register">Register</Link>
                      </Button>
                    </DropdownMenuItem>
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default Navbar;
